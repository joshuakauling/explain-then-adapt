import unittest
from pathlib import Path

from explain_then_adapt.data_generation.records import (
    AugmentationSpec,
    read_jsonl,
)
from explain_then_adapt.data_generation.validation import validate_trace_format

ROOT = Path(__file__).resolve().parents[2]
RESOURCE_DIRECTORY = ROOT / "resources" / "training"
TRAINING_MANIFEST = ROOT / "resources" / "data_generation" / "task_manifest.jsonl"


class ReasoningTrainingResourceTests(unittest.TestCase):
    def test_final_validation_views_are_complete_and_disjoint(self) -> None:
        validation = read_jsonl(RESOURCE_DIRECTORY / "reasoning_validation.jsonl")
        augmented = read_jsonl(
            RESOURCE_DIRECTORY / "reasoning_validation_augmented.jsonl"
        )
        training_task_ids = {
            record["task_id"] for record in read_jsonl(TRAINING_MANIFEST)
        }
        validation_task_ids = {record["task_id"] for record in validation}
        augmented_task_ids = {record["task_id"] for record in augmented}

        self.assertEqual(len(training_task_ids), 624)
        self.assertEqual(len(validation), 39)
        self.assertEqual(len(augmented), 39)
        self.assertEqual(validation_task_ids, augmented_task_ids)
        self.assertTrue(validation_task_ids.isdisjoint(training_task_ids))
        self.assertEqual(len({record["variant_id"] for record in augmented}), 39)

    def test_validation_traces_and_augmentations_are_structurally_valid(self) -> None:
        validation = read_jsonl(RESOURCE_DIRECTORY / "reasoning_validation.jsonl")
        augmented = read_jsonl(
            RESOURCE_DIRECTORY / "reasoning_validation_augmented.jsonl"
        )

        for record in validation:
            self.assertEqual(record["schema_version"], 1)
            self.assertEqual(record["split"], "validation")
            self.assertNotIn("augmentation", record)
            self.assertTrue(validate_trace_format(record["trace"]).accepted)

        for record in augmented:
            self.assertEqual(record["schema_version"], 1)
            self.assertEqual(record["split"], "validation_augmented")
            AugmentationSpec.from_dict(
                {
                    "source_trace_id": record["task_id"],
                    **record["augmentation"],
                }
            )
            self.assertTrue(validate_trace_format(record["trace"]).accepted)


if __name__ == "__main__":
    unittest.main()
