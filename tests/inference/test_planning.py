import json
import tempfile
import unittest
from pathlib import Path

from explain_then_adapt.inference.config import load_inference_config
from explain_then_adapt.inference.planning import (
    build_prediction_requests,
    build_reasoning_requests,
    candidate_count_by_test_input,
    create_augmentation_plan,
    load_augmentation_plan,
    protocol_needs_augmentation_plan,
    save_augmentation_plan,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "inference" / "inference.yaml"

TASK = {
    "train": [
        {"input": [[1, 0]], "output": [[2, 0]]},
        {"input": [[3]], "output": [[4]]},
    ],
    "test": [{"input": [[1]]}, {"input": [[3]]}],
}


def _write_task(directory: Path, task_id: str = "abc123") -> None:
    with (directory / f"{task_id}.json").open("w", encoding="utf-8") as file:
        json.dump(TASK, file)


class InferencePlanningTests(unittest.TestCase):
    def test_all_three_protocols_preserve_candidate_budgets(self) -> None:
        config = load_inference_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temporary_directory:
            tasks_directory = Path(temporary_directory)
            _write_task(tasks_directory)
            plan = create_augmentation_plan(
                config=config,
                task_ids=["abc123"],
                tasks_directory=tasks_directory,
            )

            standard_rm = build_reasoning_requests(
                config=config,
                task_ids=["abc123"],
                protocol="standard32",
            )
            standard_pm = build_prediction_requests(
                config=config,
                task_ids=["abc123"],
                tasks_directory=tasks_directory,
                protocol="standard32",
            )
            self.assertEqual(len(standard_rm), 1)
            self.assertEqual(len(standard_pm), 2)
            self.assertEqual({r.sample_count for r in standard_pm}, {32})
            self.assertEqual(
                set(candidate_count_by_test_input(standard_pm).values()),
                {32},
            )

            augmented_rm = build_reasoning_requests(
                config=config,
                task_ids=["abc123"],
                protocol="augmented64",
                augmentation_plan=plan,
            )
            augmented_pm = build_prediction_requests(
                config=config,
                task_ids=["abc123"],
                tasks_directory=tasks_directory,
                protocol="augmented64",
                augmentation_plan=plan,
            )
            self.assertEqual(len(augmented_rm), 64)
            self.assertEqual(len(augmented_pm), 128)
            self.assertEqual({r.sample_count for r in augmented_pm}, {1})
            self.assertEqual(
                set(candidate_count_by_test_input(augmented_pm).values()),
                {64},
            )

            budget_rm = build_reasoning_requests(
                config=config,
                task_ids=["abc123"],
                protocol="budgeted64",
                guidance_budget=8,
                augmentation_plan=plan,
            )
            budget_pm = build_prediction_requests(
                config=config,
                task_ids=["abc123"],
                tasks_directory=tasks_directory,
                protocol="budgeted64",
                guidance_budget=8,
                augmentation_plan=plan,
            )
            self.assertEqual(len(budget_rm), 8)
            self.assertEqual(
                [request.variant.variant_index for request in budget_rm],
                [7, 15, 23, 31, 39, 47, 55, 63],
            )
            self.assertEqual(len(budget_pm), 16)
            self.assertEqual({r.sample_count for r in budget_pm}, {8})
            self.assertEqual(
                set(candidate_count_by_test_input(budget_pm).values()),
                {64},
            )

            for guidance_budget in (16, 32, 64):
                rm_requests = build_reasoning_requests(
                    config=config,
                    task_ids=["abc123"],
                    protocol="budgeted64",
                    guidance_budget=guidance_budget,
                    augmentation_plan=plan,
                )
                pm_requests = build_prediction_requests(
                    config=config,
                    task_ids=["abc123"],
                    tasks_directory=tasks_directory,
                    protocol="budgeted64",
                    guidance_budget=guidance_budget,
                    augmentation_plan=plan,
                )
                self.assertEqual(len(rm_requests), guidance_budget)
                self.assertEqual(
                    {request.sample_count for request in pm_requests},
                    {64 // guidance_budget},
                )
                self.assertEqual(
                    set(candidate_count_by_test_input(pm_requests).values()),
                    {64},
                )

            zero_rm = build_reasoning_requests(
                config=config,
                task_ids=["abc123"],
                protocol="budgeted64",
                guidance_budget=0,
            )
            zero_pm = build_prediction_requests(
                config=config,
                task_ids=["abc123"],
                tasks_directory=tasks_directory,
                protocol="budgeted64",
                guidance_budget=0,
            )
            self.assertEqual(len(zero_rm), 1)
            self.assertEqual(len(zero_pm), 2)
            self.assertEqual({r.sample_count for r in zero_pm}, {64})

    def test_plan_is_saved_once_and_reloaded_for_the_same_tasks(self) -> None:
        config = load_inference_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            _write_task(directory)
            plan = create_augmentation_plan(
                config=config,
                task_ids=["abc123"],
                tasks_directory=directory,
            )
            path = directory / "augmentation_plan.json"
            save_augmentation_plan(config=config, plan=plan, output_path=path)
            loaded = load_augmentation_plan(
                config=config,
                path=path,
                task_ids=["abc123"],
                tasks_directory=directory,
            )
            self.assertEqual(
                [value.to_dict() for value in loaded["abc123"]],
                [value.to_dict() for value in plan["abc123"]],
            )
            with self.assertRaises(FileExistsError):
                save_augmentation_plan(config=config, plan=plan, output_path=path)

    def test_plan_requirement_depends_on_protocol_and_budget(self) -> None:
        self.assertFalse(protocol_needs_augmentation_plan("standard32", None))
        self.assertTrue(protocol_needs_augmentation_plan("augmented64", None))
        self.assertFalse(protocol_needs_augmentation_plan("budgeted64", 0))
        self.assertTrue(protocol_needs_augmentation_plan("budgeted64", 8))


if __name__ == "__main__":
    unittest.main()
