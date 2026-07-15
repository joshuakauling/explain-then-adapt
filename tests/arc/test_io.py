import json
import tempfile
import unittest
from pathlib import Path

from explain_then_adapt.arc.io import (
    load_existing_list,
    load_full_puzzle,
    load_puzzle_test,
    load_puzzle_train,
    load_records,
    load_subset,
    load_task,
)


class IoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

        self.task = {
            "train": [{"input": [[0]], "output": [[1]]}],
            "test": [{"input": [[2]]}],
        }
        (self.root / "abc123.json").write_text(
            json.dumps(self.task),
            encoding="utf-8",
        )
        (self.root / "small.json").write_text(
            json.dumps(["abc123.json", "def456.json"]),
            encoding="utf-8",
        )
        (self.root / "record.json").write_text(
            json.dumps({"status": "accepted"}),
            encoding="utf-8",
        )

    def test_task_sections_and_full_puzzle_are_loaded(self) -> None:
        self.assertEqual(load_task("abc123", self.root), self.task)
        self.assertEqual(load_puzzle_train("abc123", self.root), self.task["train"])
        self.assertEqual(load_puzzle_test("abc123", self.root), self.task["test"])
        self.assertEqual(
            load_full_puzzle("abc123", self.root),
            [*self.task["train"], *self.task["test"]],
        )

    def test_subset_and_generic_record_are_loaded(self) -> None:
        self.assertEqual(load_subset("small", self.root), ["abc123", "def456"])
        self.assertEqual(load_records("record", self.root), {"status": "accepted"})

    def test_missing_existing_list_defaults_to_empty(self) -> None:
        self.assertEqual(load_existing_list(self.root / "missing.json"), [])


if __name__ == "__main__":
    unittest.main()
