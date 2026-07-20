import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from explain_then_adapt.inference.artifacts import read_jsonl
from explain_then_adapt.inference.config import load_inference_config
from explain_then_adapt.inference.planning import create_augmentation_plan
from explain_then_adapt.inference.vllm_runner import (
    run_prediction_inference,
    run_reasoning_inference,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "inference" / "inference.yaml"

TASK = {
    "train": [
        {"input": [[1, 0]], "output": [[2, 0]]},
        {"input": [[3]], "output": [[4]]},
    ],
    "test": [{"input": [[1]]}],
}

GUIDANCE = (
    "General natural language description:\nReplace one color.\n\n"
    "General steps:\n1. Find the color.\n2. Replace it."
)
TRACE = f"<think>analysis</think>\n{GUIDANCE}"


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("inference fallback must not add special tokens")
        return list(text.encode("utf-8"))


class FakeCompletion:
    def __init__(self, text):
        self.text = text
        self.token_ids = list(text.encode("utf-8"))
        self.finish_reason = "stop"
        self.stop_reason = None


class FakeRequestOutput:
    def __init__(self, prompt, texts):
        self.prompt = prompt
        self.prompt_token_ids = list(prompt.encode("utf-8"))
        self.outputs = [FakeCompletion(text) for text in texts]


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeLoRARequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tokenizer = FakeTokenizer()

    def get_tokenizer(self):
        return self.tokenizer

    def chat(self, conversations, sampling_params, use_tqdm):
        self.assert_single_sample(sampling_params)
        return [
            FakeRequestOutput(value[0]["content"], [TRACE]) for value in conversations
        ]

    def generate(self, prompts, sampling_params, use_tqdm, lora_request=None):
        return [
            FakeRequestOutput(prompt, ["2"] * sampling_params.n) for prompt in prompts
        ]

    @staticmethod
    def assert_single_sample(sampling_params):
        if sampling_params.n != 1:
            raise AssertionError("RM must request exactly one sample")


def _fake_vllm_types():
    return FakeLLM, FakeSamplingParams, FakeLoRARequest


class VLLMRunnerTests(unittest.TestCase):
    def test_budgeted_run_produces_eight_by_eight_candidates(self) -> None:
        config = load_inference_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            tasks_directory = directory / "tasks"
            tasks_directory.mkdir()
            with (tasks_directory / "abc123.json").open("w", encoding="utf-8") as file:
                json.dump(TASK, file)

            plan = create_augmentation_plan(
                config=config,
                task_ids=["abc123"],
                tasks_directory=tasks_directory,
            )
            guidance_path = directory / "guidance.jsonl"
            with patch(
                "explain_then_adapt.inference.vllm_runner._vllm_types",
                _fake_vllm_types,
            ):
                rm_summary = run_reasoning_inference(
                    config=config,
                    task_ids=["abc123"],
                    tasks_directory=tasks_directory,
                    protocol="budgeted64",
                    guidance_budget=8,
                    augmentation_plan=plan,
                    model="fake-rm",
                    output_path=guidance_path,
                )
            self.assertEqual(rm_summary["request_count"], 8)
            self.assertEqual(len(list(read_jsonl(guidance_path))), 8)

            adapter_root = directory / "adapters"
            adapter = adapter_root / "abc123"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            (adapter / "ttt_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "ttt_task_adapter",
                        "task_id": "abc123",
                        "optimizer_updates": 64,
                    }
                ),
                encoding="utf-8",
            )

            prediction_path = directory / "predictions.jsonl"
            with patch(
                "explain_then_adapt.inference.vllm_runner._vllm_types",
                _fake_vllm_types,
            ):
                pm_summary = run_prediction_inference(
                    config=config,
                    task_ids=["abc123"],
                    tasks_directory=tasks_directory,
                    protocol="budgeted64",
                    guidance_mode="guided",
                    guidance_budget=8,
                    augmentation_plan=plan,
                    guidance_path=guidance_path,
                    model="fake-pm",
                    ttt_adapter_root=adapter_root,
                    output_path=prediction_path,
                )
            records = list(read_jsonl(prediction_path))
            self.assertEqual(pm_summary["request_count"], 8)
            self.assertEqual(pm_summary["candidates_per_test_input"], 64)
            self.assertEqual(pm_summary["total_candidates"], 64)
            self.assertEqual(len(records), 8)
            self.assertEqual({record["sample_count"] for record in records}, {8})
            self.assertTrue(all(record["adapter_used"] for record in records))
            self.assertTrue(
                all(record["variant"]["is_augmented"] for record in records)
            )


if __name__ == "__main__":
    unittest.main()
