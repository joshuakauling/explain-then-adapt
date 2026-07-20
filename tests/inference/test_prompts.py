import json
import tempfile
import unittest
from pathlib import Path

from explain_then_adapt.inference.config import load_inference_config
from explain_then_adapt.inference.planning import (
    InferenceVariant,
    PredictionRequest,
    ReasoningRequest,
)
from explain_then_adapt.inference.prompts import (
    build_prediction_prompt_from_task,
    build_reasoning_prompt_from_task,
    extract_guidance,
    load_guidance,
    require_guidance_for_variants,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "inference" / "inference.yaml"

TASK = {
    "train": [
        {"input": [[1, 0]], "output": [[2, 0]]},
        {"input": [[3]], "output": [[4]]},
    ],
    "test": [{"input": [[1, 3]]}],
}

GUIDANCE = (
    "General natural language description:\nReplace the marked color.\n\n"
    "General steps:\n1. Find it.\n2. Replace it."
)
TRACE = f"<think>analysis</think>\n{GUIDANCE}"


class InferencePromptTests(unittest.TestCase):
    def test_rm_and_pm_use_transformed_reordered_demonstrations(self) -> None:
        config = load_inference_config(CONFIG_PATH)
        variant = InferenceVariant(
            task_id="abc123",
            transformation_code="ID",
            value_mapping="1023456789",
            order_mapping="10",
            variant_index=0,
        )
        rm_request = ReasoningRequest(request_id=variant.key, variant=variant)
        self.assertEqual(
            build_reasoning_prompt_from_task(rm_request, TASK, config),
            "Example 1\nInput:\n3\nOutput:\n4\n\n" "Example 2\nInput:\n01\nOutput:\n21",
        )

        pm_request = PredictionRequest(
            request_id=f"{variant.key}__0",
            variant=variant,
            test_index=0,
            sample_count=1,
        )
        expected = (
            f"<|im_start|>system\n{GUIDANCE}<|im_end|>\n"
            "<|im_start|>user\n3<|im_end|>\n"
            "<|im_start|>assistant\n4<|im_end|>\n"
            "<|im_start|>user\n01<|im_end|>\n"
            "<|im_start|>assistant\n21<|im_end|>\n"
            "<|im_start|>user\n03<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.assertEqual(
            build_prediction_prompt_from_task(
                pm_request,
                TASK,
                config,
                guidance=GUIDANCE,
            ),
            expected,
        )
        unguided = build_prediction_prompt_from_task(
            pm_request,
            TASK,
            config,
            guidance=None,
        )
        self.assertTrue(unguided.startswith("<|im_start|>user\n3"))
        self.assertNotIn("<|im_start|>system", unguided)

    def test_guidance_loader_supports_new_jsonl_and_legacy_index_zero(self) -> None:
        self.assertEqual(extract_guidance(TRACE), GUIDANCE)
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            jsonl_path = directory / "guidance.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        {
                            "kind": "reasoning_guidance",
                            "request_id": "abc123",
                            "raw_output": TRACE,
                            "guidance": GUIDANCE,
                        }
                    )
                    + "\n"
                )
            self.assertEqual(load_guidance(jsonl_path), {"abc123": GUIDANCE})

            legacy_path = directory / "legacy.json"
            with legacy_path.open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        "meta": {"sampling_size": 8},
                        "predictions": {"abc123": [TRACE, "unused"]},
                    },
                    file,
                )
            self.assertEqual(load_guidance(legacy_path), {"abc123": GUIDANCE})

    def test_missing_exact_variant_guidance_fails_before_inference(self) -> None:
        variant = InferenceVariant(task_id="abc123")
        with self.assertRaisesRegex(ValueError, "missing 1"):
            require_guidance_for_variants([variant], {})


if __name__ == "__main__":
    unittest.main()
