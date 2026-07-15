import json
import tempfile
import unittest
from pathlib import Path

from explain_then_adapt.data_generation.hints import (
    HintStatus,
    load_hint_file,
    load_hints_jsonl,
)


COMPLETE_HINT = {
    "general": "General observation",
    "inputs": "Input structures",
    "outputs": "Output structures",
    "transformation": "Transformation rule",
    "transformation_steps": "Apply the rule",
}


class HintTests(unittest.TestCase):
    def test_legacy_list_wrapped_hint_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            path.write_text(json.dumps([COMPLETE_HINT]), encoding="utf-8")

            result = load_hint_file(path)

        self.assertEqual(result.status, HintStatus.COMPLETE)
        self.assertIsNotNone(result.hint)
        self.assertIn("Transformation Steps:\nApply the rule", result.hint.format())

    def test_missing_and_partial_hints_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = load_hint_file(Path(directory) / "missing.json")
            path = Path(directory) / "partial.json"
            partial_value = dict(COMPLETE_HINT)
            partial_value["outputs"] = " "
            path.write_text(json.dumps(partial_value), encoding="utf-8")
            partial = load_hint_file(path)

        self.assertEqual(missing.status, HintStatus.MISSING)
        self.assertEqual(partial.status, HintStatus.INCOMPLETE)
        self.assertEqual(partial.missing_fields, ("outputs",))
        self.assertIsNone(partial.hint)

    def test_versioned_jsonl_hints_are_loaded_by_task_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hints.jsonl"
            records = [
                {"schema_version": 1, "task_id": "task_a", **COMPLETE_HINT},
                {
                    "schema_version": 1,
                    "task_id": "task_b",
                    **{**COMPLETE_HINT, "general": "Second observation"},
                },
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            hints = load_hints_jsonl(path)

        self.assertEqual(set(hints), {"task_a", "task_b"})
        self.assertEqual(hints["task_b"].general, "Second observation")

    def test_duplicate_jsonl_hint_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hints.jsonl"
            record = {"schema_version": 1, "task_id": "task", **COMPLETE_HINT}
            path.write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate hint task_id"):
                load_hints_jsonl(path)


if __name__ == "__main__":
    unittest.main()
