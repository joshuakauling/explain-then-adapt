import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from explain_then_adapt.training.config import load_ttt_training_config
from explain_then_adapt.training.prediction_trainer import PredictionAssistantCollator
from explain_then_adapt.training.ttt_data import (
    build_ttt_records,
    generate_ttt_augmentation_plan,
    load_ttt_augmentation_plan,
    normalize_ttt_task_ids,
    resolve_ttt_guidance,
    selected_guidance_variant_indices,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "training" / "test_time_training.yaml"

TASK = {
    "train": [
        {"input": [[1, 0], [0, 1]], "output": [[2, 0], [0, 2]]},
        {"input": [[1, 1]], "output": [[2, 2]]},
    ],
    "test": [{"input": [[1]]}],
}

GUIDANCE = (
    "General natural language description:\nReplace one color.\n\n"
    "General steps:\n1. Find the source color.\n2. Replace it."
)


class FakeTokenizer:
    name_or_path = "fake-qwen"
    vocab_size = 256
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"

    def encode(self, text: str, *, add_special_tokens: bool):
        if add_special_tokens:
            raise AssertionError("TTT serialization must disable special tokens")
        return list(text.encode("utf-8"))


def _write_task(directory: Path, task_id: str) -> None:
    with (directory / f"{task_id}.json").open("w", encoding="utf-8") as file:
        json.dump(TASK, file)


class TTTAugmentationPlanTests(unittest.TestCase):
    def test_task_ids_are_normalized_before_plan_use(self) -> None:
        self.assertEqual(
            normalize_ttt_task_ids(["abc123.json", "def456"]),
            ["abc123", "def456"],
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalize_ttt_task_ids(["abc123", "abc123.json"])

    def test_inference_jsonl_guidance_is_accepted(self) -> None:
        from explain_then_adapt.training.ttt_data import load_ttt_guidance

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "guidance.jsonl"
            augmented_key = "abc123_ID_1023456789_10"
            with path.open("w", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        {
                            "kind": "reasoning_guidance",
                            "request_id": augmented_key,
                            "guidance": GUIDANCE,
                        }
                    )
                    + "\n"
                )
            self.assertEqual(
                load_ttt_guidance(path),
                {augmented_key: GUIDANCE},
            )

    def test_plan_is_stable_balanced_and_replayable_from_legacy_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            _write_task(directory, "abc123")
            _write_task(directory, "def456")

            first = generate_ttt_augmentation_plan(
                task_ids=["def456", "abc123"],
                tasks_directory=directory,
                seed=167,
            )
            second = generate_ttt_augmentation_plan(
                task_ids=["abc123", "def456"],
                tasks_directory=directory,
                seed=167,
            )
            self.assertEqual(
                {
                    key: [variant.to_dict() for variant in values]
                    for key, values in first.items()
                },
                {
                    key: [variant.to_dict() for variant in values]
                    for key, values in second.items()
                },
            )
            for variants in first.values():
                self.assertEqual(len(variants), 64)
                self.assertEqual(
                    [variant.variant_index for variant in variants],
                    list(range(64)),
                )
                self.assertEqual(
                    {
                        code: sum(v.transformation_code == code for v in variants)
                        for code in {v.transformation_code for v in variants}
                    },
                    {code: 8 for code in {v.transformation_code for v in variants}},
                )

            legacy_path = directory / "legacy-plan.json"
            with legacy_path.open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        task_id: [variant.augmented_key for variant in variants]
                        for task_id, variants in first.items()
                    },
                    file,
                )
            replayed = load_ttt_augmentation_plan(
                legacy_path,
                task_ids=["abc123", "def456"],
                tasks_directory=directory,
            )
            self.assertEqual(
                {
                    key: [variant.to_dict() for variant in values]
                    for key, values in replayed.items()
                },
                {
                    key: [variant.to_dict() for variant in values]
                    for key, values in first.items()
                },
            )

    def test_budget_subsets_are_balanced_and_nested(self) -> None:
        selections = {
            budget: set(selected_guidance_variant_indices(budget))
            for budget in (0, 8, 16, 32, 64)
        }
        self.assertEqual(
            [len(selections[value]) for value in selections], [0, 8, 16, 32, 64]
        )
        self.assertTrue(selections[0] <= selections[8])
        self.assertTrue(selections[8] <= selections[16])
        self.assertTrue(selections[16] <= selections[32])
        self.assertTrue(selections[32] <= selections[64])
        self.assertEqual(
            sorted(index for index in selections[16] if index < 8),
            [3, 7],
        )


class TTTGuidanceAndTokenizationTests(unittest.TestCase):
    def test_guidance_states_keep_budget_empty_distinct_from_missing(self) -> None:
        config = load_ttt_training_config(
            CONFIG_PATH,
            "guided",
            guidance_budget=8,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            _write_task(directory, "abc123")
            variants = generate_ttt_augmentation_plan(
                task_ids=["abc123"],
                tasks_directory=directory,
                seed=config.seed,
            )["abc123"]
            unselected = variants[0]
            selected = variants[7]

            empty = resolve_ttt_guidance(
                unselected,
                config=config,
                guidance_by_key={},
            )
            self.assertEqual(empty.status, "budget_empty")
            self.assertEqual(empty.content, " ")

            provided = resolve_ttt_guidance(
                selected,
                config=config,
                guidance_by_key={selected.augmented_key: GUIDANCE},
            )
            self.assertEqual(provided.status, "provided")
            self.assertEqual(provided.content, GUIDANCE)
            with self.assertRaisesRegex(ValueError, "has no guidance"):
                resolve_ttt_guidance(
                    selected,
                    config=config,
                    guidance_by_key={},
                )

            compatibility = replace(
                config,
                data=replace(config.data, missing_guidance_policy="omit_system"),
            )
            missing = resolve_ttt_guidance(
                selected,
                config=compatibility,
                guidance_by_key={},
            )
            self.assertEqual(missing.status, "missing")
            self.assertIsNone(missing.content)

    def test_budget_zero_uses_whitespace_system_and_masks_first_output(self) -> None:
        config = load_ttt_training_config(
            CONFIG_PATH,
            "guided",
            guidance_budget=0,
        )
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            _write_task(directory, "abc123")
            plan = generate_ttt_augmentation_plan(
                task_ids=["abc123"],
                tasks_directory=directory,
                seed=config.seed,
            )
            records = build_ttt_records(
                config=config,
                tokenizer=tokenizer,
                tasks_directory=directory,
                task_ids=["abc123"],
                augmentation_plan=plan,
                guidance_by_key={},
            )["abc123"]

            self.assertEqual(len(records), 64)
            self.assertEqual(
                {record["guidance_status"] for record in records}, {"budget_empty"}
            )
            text = bytes(records[0]["input_ids"].tolist()).decode("utf-8")
            self.assertTrue(text.startswith("<|im_start|>system\n <|im_end|>\n"))
            self.assertTrue(text.endswith("<|im_end|>\n"))

            batch = PredictionAssistantCollator(
                pad_token_id=tokenizer.pad_token_id,
                ignore_first_response=True,
                pad_to_multiple_of=1,
            )([records[0]])
            first_start, first_end = records[0]["assistant_spans"][0]
            second_start, second_end = records[0]["assistant_spans"][1]
            self.assertTrue((batch["labels"][0, first_start:first_end] == -100).all())
            self.assertTrue((batch["labels"][0, second_start:second_end] != -100).all())

            unguided = load_ttt_training_config(CONFIG_PATH, "unguided")
            unguided_record = build_ttt_records(
                config=unguided,
                tokenizer=tokenizer,
                tasks_directory=directory,
                task_ids=["abc123"],
                augmentation_plan=plan,
                guidance_by_key={},
            )["abc123"][0]
            unguided_text = bytes(unguided_record["input_ids"].tolist()).decode("utf-8")
            self.assertFalse(unguided_text.startswith("<|im_start|>system\n"))
            self.assertEqual(unguided_record["guidance_status"], "unguided")


if __name__ == "__main__":
    unittest.main()
