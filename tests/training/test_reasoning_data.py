import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch

from explain_then_adapt.data_generation.records import (
    AugmentationSpec,
    ChatMessage,
    GenerationRequest,
    GenerationResult,
    GenerationStage,
    write_jsonl,
)
from explain_then_adapt.training.config import load_reasoning_training_config
from explain_then_adapt.training.reasoning_data import build_reasoning_token_cache
from explain_then_adapt.training.reasoning_trainer import (
    validate_reasoning_cache_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "training" / "reasoning_model.yaml"
TRACE = """<think>
1) INPUT ANALYSIS
The inputs contain a colored cell.
2) OUTPUT ANALYSIS
The outputs move that cell.
3) TRANSFORMATION ANALYSIS
Move the nonzero cell one column to the right.
4) STEPS FOR THE TRANSFORMATION
Locate the nonzero cell and move it one column to the right.
</think>
General natural language description:
Move the nonzero cell one column to the right.
General steps:
1. Locate the nonzero cell.
2. Move it one column to the right."""
TASK = {
    "train": [
        {"input": [[1, 0]], "output": [[0, 1]]},
        {"input": [[2, 0]], "output": [[0, 2]]},
    ],
    "test": [{"input": [[9, 9]]}],
}
VALIDATION_TASK = {
    "train": [
        {"input": [[3, 0]], "output": [[0, 3]]},
        {"input": [[4, 0]], "output": [[0, 4]]},
    ],
    "test": [{"input": [[8, 8]]}],
}


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    name_or_path = "fake-qwen"
    init_kwargs: Mapping[str, str] = {"_commit_hash": "fake-revision"}

    def __init__(self) -> None:
        self.prompts: List[str] = []

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: Any,
    ) -> List[int]:
        self.assert_contract(messages, tokenize, add_generation_prompt, return_tensors)
        self.prompts.append(messages[0]["content"])
        assistant_header = list("<|im_start|>assistant\n".encode("utf-8"))
        assistant_target = list(messages[1]["content"].encode("utf-8"))
        return [10, 11, *assistant_header, *assistant_target, self.eos_token_id]

    @staticmethod
    def assert_contract(
        messages: Sequence[Mapping[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: Any,
    ) -> None:
        if (
            len(messages) != 2
            or messages[0]["role"] != "user"
            or messages[1]["role"] != "assistant"
            or not tokenize
            or add_generation_prompt
            or return_tensors is not None
        ):
            raise AssertionError("unexpected chat-template invocation")

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool) -> List[int]:
        if add_special_tokens:
            raise AssertionError("assistant header must not add special tokens")
        return list(text.encode("utf-8"))


def _write_task(path: Path, task: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(task), encoding="utf-8")


class ReasoningDataBuilderTests(unittest.TestCase):
    def test_builds_deterministic_task_balanced_cache(self) -> None:
        base_config = load_reasoning_training_config(CONFIG_PATH)
        config = replace(
            base_config,
            data=replace(base_config.data, variants_per_task=2),
            optimization=replace(base_config.optimization, epochs=2),
            control=replace(base_config.control, checkpoint_epochs=(1, 2)),
        )
        tokenizer = FakeTokenizer()

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            tasks_directory = directory / "tasks"
            tasks_directory.mkdir()
            _write_task(tasks_directory / "train-task.json", TASK)
            _write_task(tasks_directory / "validation-task.json", VALIDATION_TASK)

            task_manifest = directory / "task_manifest.jsonl"
            write_jsonl([{"task_id": "train-task"}], task_manifest)

            requests: List[GenerationRequest] = []
            results: List[GenerationResult] = []
            mappings = ("0123456789", "0213456789", "0321456789")
            for index, value_mapping in enumerate(mappings):
                spec = AugmentationSpec(
                    source_trace_id="train-task",
                    transformation_code="ID",
                    value_mapping=value_mapping,
                    order_mapping="01",
                    variant_index=index,
                )
                request_id = f"rewrite-train-task-{index}"
                requests.append(
                    GenerationRequest(
                        request_id=request_id,
                        task_id="train-task",
                        stage=GenerationStage.REWRITE,
                        messages=(ChatMessage("user", "Rewrite the trace."),),
                        prompt_version="test",
                        augmentation=spec,
                    )
                )
                results.append(
                    GenerationResult(
                        request_id=request_id,
                        task_id="train-task",
                        stage=GenerationStage.REWRITE,
                        backend="test",
                        model="test",
                        augmentation=spec,
                        raw_output=TRACE,
                        normalized_output=TRACE if index else None,
                        validation={"static": {"accepted": True}},
                    )
                )

            request_path = directory / "rewrite_requests.jsonl"
            result_path = directory / "rewrite_results.jsonl"
            write_jsonl((request.to_dict() for request in requests), request_path)
            write_jsonl((result.to_dict() for result in results), result_path)

            validation_path = directory / "validation.jsonl"
            augmented_validation_path = directory / "validation_augmented.jsonl"
            write_jsonl(
                [
                    {
                        "schema_version": 1,
                        "split": "validation",
                        "variant_id": "validation-task",
                        "task_id": "validation-task",
                        "trace": TRACE,
                    }
                ],
                validation_path,
            )
            write_jsonl(
                [
                    {
                        "schema_version": 1,
                        "split": "validation_augmented",
                        "variant_id": ("validation-task_ID_0123456789_01"),
                        "task_id": "validation-task",
                        "augmentation": {
                            "transformation_code": "ID",
                            "value_mapping": "0123456789",
                            "order_mapping": "01",
                        },
                        "trace": TRACE,
                    }
                ],
                augmented_validation_path,
            )

            output_cache = directory / "reasoning.pt"
            output_manifest = directory / "reasoning_manifest.json"
            manifest = build_reasoning_token_cache(
                config=config,
                tokenizer=tokenizer,
                tasks_directory=tasks_directory,
                task_manifest_path=task_manifest,
                rewrite_request_paths=[request_path],
                rewrite_result_paths=[result_path],
                validation_path=validation_path,
                augmented_validation_path=augmented_validation_path,
                output_cache_path=output_cache,
                output_manifest_path=output_manifest,
            )

            self.assertEqual(manifest["counts"]["accepted_results"], 3)
            self.assertEqual(manifest["counts"]["train_tasks"], 1)
            self.assertEqual(manifest["counts"]["train_variants"], 2)
            self.assertEqual(
                manifest["valid_candidates_per_task"],
                {"train-task": 3},
            )
            cache = torch.load(output_cache, map_location="cpu", weights_only=False)
            validate_reasoning_cache_config(cache, config)
            with self.assertRaisesRegex(ValueError, "data settings"):
                validate_reasoning_cache_config(
                    cache,
                    replace(config, data=replace(config.data, variants_per_task=3)),
                )
            records: List[Dict[str, Any]] = cache["splits"]["train"]["train-task"]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["input_ids"].dtype, torch.int32)
            assistant_start = 2 + len(config.data.assistant_header.encode("utf-8"))
            target_tokens = list(TRACE.encode("utf-8"))
            for record in records:
                self.assertEqual(record["assistant_start"], assistant_start)
                self.assertEqual(
                    record["input_ids"][assistant_start:-1].tolist(),
                    target_tokens,
                )
                self.assertEqual(record["input_ids"][-1].item(), tokenizer.eos_token_id)

            selected_ids = [record["variant_id"] for record in records]
            second_cache = directory / "reasoning-second.pt"
            build_reasoning_token_cache(
                config=config,
                tokenizer=FakeTokenizer(),
                tasks_directory=tasks_directory,
                task_manifest_path=task_manifest,
                rewrite_request_paths=[request_path],
                rewrite_result_paths=[result_path],
                validation_path=validation_path,
                augmented_validation_path=augmented_validation_path,
                output_cache_path=second_cache,
                output_manifest_path=directory / "reasoning-second.json",
            )
            second = torch.load(second_cache, map_location="cpu", weights_only=False)
            self.assertEqual(
                selected_ids,
                [
                    record["variant_id"]
                    for record in second["splits"]["train"]["train-task"]
                ],
            )

            self.assertTrue(tokenizer.prompts)
            self.assertTrue(all("99" not in prompt for prompt in tokenizer.prompts))


if __name__ == "__main__":
    unittest.main()
