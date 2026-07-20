import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, List, Mapping

import torch

from explain_then_adapt.data_generation.records import (
    AugmentationSpec,
    ChatMessage,
    GenerationRequest,
    GenerationResult,
    GenerationStage,
    write_jsonl,
)
from explain_then_adapt.training.config import (
    PredictionTrainingConfig,
    load_prediction_training_config,
)
from explain_then_adapt.training.prediction_data import (
    REARC_REVISION,
    build_prediction_token_cache,
    extract_prediction_guidance,
)
from explain_then_adapt.training.prediction_trainer import (
    validate_prediction_cache_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "training" / "prediction_model.yaml"
TRACE = """<think>
1) INPUT ANALYSIS
The input contains a marker.
2) OUTPUT ANALYSIS
The output moves the marker.
3) TRANSFORMATION ANALYSIS
Move the nonzero value one column right.
4) STEPS FOR THE TRANSFORMATION
Locate the marker and move it right.
</think>
General natural language description:
Move the nonzero value one column to the right.
General steps:
1. Locate the nonzero value.
2. Move it one column to the right."""
TRAIN_TASK = {
    "train": [
        {"input": [[1, 0]], "output": [[0, 1]]},
        {"input": [[2, 0]], "output": [[0, 2]]},
    ],
    "test": [{"input": [[3, 0]], "output": [[0, 3]]}],
}
VALIDATION_TASK = {
    "train": [
        {"input": [[4, 0]], "output": [[0, 4]]},
        {"input": [[5, 0]], "output": [[0, 5]]},
    ],
    "test": [{"input": [[6, 0]], "output": [[0, 6]]}],
}


class FakeTokenizer:
    pad_token_id = 255
    eos_token_id = 2
    name_or_path = "fake-qwen"
    init_kwargs: Mapping[str, str] = {"_commit_hash": "fake-revision"}

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool) -> List[int]:
        if add_special_tokens:
            raise AssertionError("manual PM serialization must not add special tokens")
        return list(text.encode("utf-8"))


def _tiny_config(profile: str) -> PredictionTrainingConfig:
    base = load_prediction_training_config(CONFIG_PATH, profile)
    return replace(
        base,
        data=replace(
            base.data,
            variants_per_task=2,
            synthetic_task_count=1,
            rearc_task_count=1,
            rearc_examples_per_task=3,
            rearc_pairs_per_variant=2,
            pad_to_multiple_of=1,
        ),
        optimization=replace(
            base.optimization,
            epochs=2,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            validation_batch_size=1,
            warmup_ratio=0.0,
        ),
        control=replace(
            base.control,
            checkpoint_epochs=(),
            checkpoint_steps=(),
        ),
        loader=replace(base.loader, pin_memory=False),
    )


def _write_task(path: Path, task: Any) -> None:
    path.write_text(json.dumps(task), encoding="utf-8")


def _write_validation_resources(directory: Path) -> tuple[Path, Path]:
    validation_path = directory / "validation.jsonl"
    augmented_path = directory / "validation_augmented.jsonl"
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
                "variant_id": "validation-task_ID_0123456789_01",
                "task_id": "validation-task",
                "augmentation": {
                    "transformation_code": "ID",
                    "value_mapping": "0123456789",
                    "order_mapping": "01",
                },
                "trace": TRACE,
            }
        ],
        augmented_path,
    )
    return validation_path, augmented_path


class PredictionDataBuilderTests(unittest.TestCase):
    def test_builds_guided_synthetic_cache_without_thinking_targets(self) -> None:
        config = _tiny_config("guided")
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            tasks_directory = directory / "tasks"
            tasks_directory.mkdir()
            _write_task(tasks_directory / "train-task.json", TRAIN_TASK)
            _write_task(
                tasks_directory / "validation-task.json",
                VALIDATION_TASK,
            )
            validation_path, augmented_path = _write_validation_resources(directory)

            task_manifest = directory / "task_manifest.jsonl"
            write_jsonl([{"task_id": "train-task"}], task_manifest)
            requests = []
            results = []
            for index, value_mapping in enumerate(
                ("0123456789", "0213456789", "0321456789")
            ):
                augmentation = AugmentationSpec(
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
                        augmentation=augmentation,
                    )
                )
                results.append(
                    GenerationResult(
                        request_id=request_id,
                        task_id="train-task",
                        stage=GenerationStage.REWRITE,
                        backend="test",
                        model="test",
                        augmentation=augmentation,
                        raw_output=TRACE,
                        normalized_output=TRACE,
                        validation={"static": {"accepted": True}},
                    )
                )
            request_path = directory / "requests.jsonl"
            result_path = directory / "results.jsonl"
            write_jsonl((request.to_dict() for request in requests), request_path)
            write_jsonl((result.to_dict() for result in results), result_path)

            output_cache = directory / "prediction.pt"
            manifest = build_prediction_token_cache(
                config=config,
                tokenizer=tokenizer,
                tasks_directory=tasks_directory,
                task_manifest_path=task_manifest,
                rewrite_request_paths=[request_path],
                rewrite_result_paths=[result_path],
                validation_path=validation_path,
                augmented_validation_path=augmented_path,
                output_cache_path=output_cache,
                output_manifest_path=directory / "prediction.manifest.json",
            )

            self.assertEqual(manifest["counts"]["accepted_results"], 3)
            self.assertEqual(manifest["counts"]["train_variants"], 2)
            cache = torch.load(output_cache, map_location="cpu", weights_only=False)
            validate_prediction_cache_config(cache, config)
            validate_prediction_cache_config(
                cache,
                _tiny_config("guided_see_first"),
            )
            validate_prediction_cache_config(cache, _tiny_config("guided_rearc"))
            with self.assertRaisesRegex(ValueError, "settings do not match"):
                validate_prediction_cache_config(cache, _tiny_config("unguided"))

            records = cache["splits"]["train"]["train-task"]
            self.assertEqual(len(records), 2)
            for record in records:
                text = bytes(record["input_ids"].tolist()).decode("utf-8")
                self.assertIn("<|im_start|>system\n", text)
                self.assertIn("General natural language description:", text)
                self.assertNotIn("<think>", text)
                self.assertNotIn("INPUT ANALYSIS", text)
                self.assertEqual(len(record["assistant_spans"]), 3)
                self.assertEqual(record["n_pairs_train"], 2)
                self.assertEqual(record["n_pairs_test"], 1)
                for start, end in record["assistant_spans"]:
                    grid = bytes(record["input_ids"][start:end].tolist()).decode(
                        "utf-8"
                    )
                    self.assertRegex(grid, r"^[0-9]+(?:\n[0-9]+)*$")

            second_cache = directory / "prediction-second.pt"
            build_prediction_token_cache(
                config=config,
                tokenizer=FakeTokenizer(),
                tasks_directory=tasks_directory,
                task_manifest_path=task_manifest,
                rewrite_request_paths=[request_path],
                rewrite_result_paths=[result_path],
                validation_path=validation_path,
                augmented_validation_path=augmented_path,
                output_cache_path=second_cache,
                output_manifest_path=directory / "prediction-second.json",
            )
            second = torch.load(second_cache, map_location="cpu", weights_only=False)
            for first_record, second_record in zip(
                records,
                second["splits"]["train"]["train-task"],
            ):
                self.assertEqual(
                    first_record["variant_id"], second_record["variant_id"]
                )
                self.assertTrue(
                    torch.equal(first_record["input_ids"], second_record["input_ids"])
                )

    def test_builds_deterministic_external_rearc_cache(self) -> None:
        config = _tiny_config("unguided_rearc")
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            tasks_directory = directory / "tasks"
            tasks_directory.mkdir()
            _write_task(
                tasks_directory / "validation-task.json",
                VALIDATION_TASK,
            )
            validation_path, augmented_path = _write_validation_resources(directory)
            rearc_directory = directory / "rearc"
            rearc_directory.mkdir()
            _write_task(
                rearc_directory / "rearc-task.json",
                [
                    {"input": [[1, 0]], "output": [[0, 1]]},
                    {"input": [[2, 0]], "output": [[0, 2]]},
                    {"input": [[3, 0]], "output": [[0, 3]]},
                ],
            )

            output_cache = directory / "rearc.pt"
            manifest = build_prediction_token_cache(
                config=config,
                tokenizer=FakeTokenizer(),
                tasks_directory=tasks_directory,
                rearc_tasks_directory=rearc_directory,
                validation_path=validation_path,
                augmented_validation_path=augmented_path,
                output_cache_path=output_cache,
                output_manifest_path=directory / "rearc.manifest.json",
            )

            self.assertEqual(manifest["counts"]["source_pairs"], 3)
            self.assertEqual(manifest["counts"]["packed_pairs"], 4)
            self.assertEqual(
                manifest["sources"]["rearc"]["expected_revision"],
                REARC_REVISION,
            )
            cache = torch.load(output_cache, map_location="cpu", weights_only=False)
            records = cache["splits"]["train"]["rearc-task"]
            self.assertEqual(len(records), 2)
            for record in records:
                self.assertFalse(record["guided"])
                self.assertEqual(len(record["assistant_spans"]), 2)
                self.assertEqual(len(record["pair_augmentations"]), 2)
                self.assertLessEqual(
                    record["sequence_length"],
                    config.data.max_sequence_length,
                )

    def test_guidance_extraction_requires_the_compact_sections(self) -> None:
        guidance = extract_prediction_guidance(TRACE)
        self.assertTrue(guidance.startswith("General natural language description:"))
        self.assertNotIn("INPUT ANALYSIS", guidance)
        with self.assertRaisesRegex(ValueError, "no guidance"):
            extract_prediction_guidance("no closing marker")


if __name__ == "__main__":
    unittest.main()
