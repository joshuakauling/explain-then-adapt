import unittest
from collections import Counter
from pathlib import Path

from explain_then_adapt.data_generation.hints import load_hints_jsonl
from explain_then_adapt.data_generation.records import read_jsonl
from explain_then_adapt.data_generation.validation import validate_trace_format


RESOURCE_DIRECTORY = Path(__file__).parents[2] / "resources" / "data_generation"


class ResourceTests(unittest.TestCase):
    def test_canonical_resource_counts_and_references(self) -> None:
        base_traces = read_jsonl(
            RESOURCE_DIRECTORY / "base_reasoning_traces.jsonl"
        )
        base_trace_repairs = read_jsonl(
            RESOURCE_DIRECTORY / "base_trace_repairs.jsonl"
        )
        hints = load_hints_jsonl(RESOURCE_DIRECTORY / "hints.jsonl")
        few_shots = read_jsonl(RESOURCE_DIRECTORY / "few_shot_traces.jsonl")
        tasks = read_jsonl(RESOURCE_DIRECTORY / "task_manifest.jsonl")

        self.assertEqual(len(base_traces), 624)
        self.assertEqual(len(base_trace_repairs), 24)
        self.assertEqual(len(hints), 481)
        self.assertEqual(len(few_shots), 5)
        self.assertEqual(len(tasks), 624)
        self.assertEqual(len({task["task_id"] for task in tasks}), 624)
        self.assertEqual(
            Counter(task["corpus_partition"] for task in tasks),
            {"legacy_training_2024": 391, "training_2025_addition": 233},
        )
        self.assertEqual(
            Counter(task["hint_mode"] for task in tasks),
            {"provided": 481, "none": 143},
        )
        self.assertEqual(
            Counter(record["repair"] for record in base_trace_repairs),
            {
                "insert_missing_think_close": 22,
                "remove_duplicate_summary_inside_think": 2,
            },
        )
        self.assertEqual(
            {record["task_id"] for record in base_traces},
            {task["task_id"] for task in tasks},
        )
        self.assertTrue(
            all(
                task["hint_resource_id"] in hints
                for task in tasks
                if task["hint_resource_id"] is not None
            )
        )

    def test_few_shot_pool_order_and_trace_format(self) -> None:
        base_traces = {
            record["task_id"]: record["trace"]
            for record in read_jsonl(
                RESOURCE_DIRECTORY / "base_reasoning_traces.jsonl"
            )
        }
        few_shots = read_jsonl(RESOURCE_DIRECTORY / "few_shot_traces.jsonl")

        self.assertEqual(
            [record["task_id"] for record in few_shots],
            ["6430c8c4", "7c008303", "08ed6ac7", "60b61512", "f25fbde4"],
        )
        self.assertTrue(
            all(validate_trace_format(record["trace"]).accepted for record in few_shots)
        )
        self.assertTrue(
            all(validate_trace_format(trace).accepted for trace in base_traces.values())
        )
        self.assertTrue(
            all(
                record["trace"] == base_traces[record["task_id"]]
                for record in few_shots
            )
        )


if __name__ == "__main__":
    unittest.main()
