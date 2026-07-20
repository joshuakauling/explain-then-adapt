import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from explain_then_adapt.training.config import load_reasoning_training_config
from explain_then_adapt.training.reasoning_trainer import (
    AssistantOnlyCollator,
    TaskVariantDataset,
    optimizer_steps_per_epoch,
    train_reasoning_model,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "training" / "reasoning_model.yaml"


class TaskVariantDatasetTests(unittest.TestCase):
    def test_uses_each_task_variant_once_across_epochs(self) -> None:
        dataset = TaskVariantDataset(
            {
                "task-a": [
                    {
                        "task_id": "task-a",
                        "variant_id": "a-0",
                        "input_ids": [1, 2],
                        "assistant_start": 1,
                    },
                    {
                        "task_id": "task-a",
                        "variant_id": "a-1",
                        "input_ids": [1, 3],
                        "assistant_start": 1,
                    },
                ],
                "task-b": [
                    {
                        "task_id": "task-b",
                        "variant_id": "b-0",
                        "input_ids": [4, 5],
                        "assistant_start": 1,
                    },
                    {
                        "task_id": "task-b",
                        "variant_id": "b-1",
                        "input_ids": [4, 6],
                        "assistant_start": 1,
                    },
                ],
            },
            seed=167,
        )

        variants_by_epoch = []
        for epoch in (0, 1):
            dataset.set_epoch(epoch)
            variants_by_epoch.append(
                {
                    dataset[index]["task_id"]: dataset[index]["variant_id"]
                    for index in range(len(dataset))
                }
            )

        self.assertEqual(
            variants_by_epoch,
            [
                {"task-a": "a-0", "task-b": "b-0"},
                {"task-a": "a-1", "task-b": "b-1"},
            ],
        )
        with self.assertRaisesRegex(ValueError, "requires at least"):
            dataset.set_epoch(2)


class AssistantOnlyCollatorTests(unittest.TestCase):
    def test_masks_user_tokens_and_right_padding(self) -> None:
        collator = AssistantOnlyCollator(
            pad_token_id=0,
            pad_to_multiple_of=4,
        )

        batch = collator(
            [
                {"input_ids": [1, 2, 3], "assistant_start": 2},
                {
                    "input_ids": torch.tensor([4, 5], dtype=torch.int32),
                    "assistant_start": 1,
                },
            ]
        )

        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 3, 0], [4, 5, 0, 0]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 0], [1, 1, 0, 0]])
        self.assertEqual(
            batch["labels"].tolist(),
            [[-100, -100, 3, -100], [-100, 5, -100, -100]],
        )

    def test_final_partial_optimizer_update_is_counted(self) -> None:
        self.assertEqual(optimizer_steps_per_epoch(624, 2, 8), 39)
        self.assertEqual(optimizer_steps_per_epoch(625, 2, 8), 40)


class TinyReasoningModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, **batch: torch.Tensor) -> SimpleNamespace:
        labels = batch["labels"]
        target = labels[labels != -100].float()
        return SimpleNamespace(loss=target.mean() + self.anchor * 0.0)

    def save_pretrained(self, directory: Path, *, safe_serialization: bool) -> None:
        if not safe_serialization:
            raise AssertionError("checkpoints must use safe serialization")
        torch.save(self.state_dict(), directory / "adapter_model.pt")


class FakeWandbRun:
    def __init__(self) -> None:
        self.logs = []
        self.finished = False

    def log(self, metrics, *, step: int) -> None:
        self.logs.append((step, dict(metrics)))

    def finish(self) -> None:
        self.finished = True


class ReasoningTrainingLoopTests(unittest.TestCase):
    def test_original_loop_order_logging_and_checkpoint_contract(self) -> None:
        base = load_reasoning_training_config(CONFIG_PATH)
        config = replace(
            base,
            model=replace(
                base.model,
                dtype="float32",
                attention_implementation="sdpa",
                gradient_checkpointing=False,
            ),
            data=replace(
                base.data,
                variants_per_task=2,
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
                log_every_steps=3,
                validate_every_steps=2,
                checkpoint_every_steps=3,
                checkpoint_epochs=(1,),
            ),
            loader=replace(base.loader, pin_memory=False),
        )

        def record(task_id: str, variant_id: str, target: int):
            return {
                "task_id": task_id,
                "variant_id": variant_id,
                "input_ids": torch.tensor([99, target], dtype=torch.int32),
                "assistant_start": 1,
            }

        cache = {
            "schema_version": 1,
            "metadata": {
                "config": config.to_dict(),
                "tokenizer": {"pad_token_id": 0},
            },
            "splits": {
                "train": {
                    "task-a": [record("task-a", "a-0", 1), record("task-a", "a-1", 5)],
                    "task-b": [record("task-b", "b-0", 3), record("task-b", "b-1", 7)],
                },
                "validation": {
                    "val": [record("val", "val", 2)],
                },
                "validation_augmented": {
                    "val": [record("val", "val-aug", 4)],
                },
            },
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            cache_path = directory / "reasoning.pt"
            torch.save(cache, cache_path)
            model = TinyReasoningModel()
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.optimization.peak_learning_rate,
            )
            wandb_run = FakeWandbRun()
            wandb_module = types.ModuleType("wandb")
            wandb_module.init = lambda **_: wandb_run  # type: ignore[attr-defined]

            with (
                patch(
                    "explain_then_adapt.training.reasoning_trainer.build_model_and_optimizer",
                    return_value=(model, optimizer),
                ),
                patch.dict("sys.modules", {"wandb": wandb_module}),
            ):
                summary = train_reasoning_model(
                    config=config,
                    cache_path=cache_path,
                    output_directory=directory / "runs",
                    run_name="test-run",
                    use_wandb=True,
                )

            run_directory = directory / "runs" / "test-run"
            self.assertEqual(summary["global_steps"], 4)
            self.assertEqual(summary["steps_per_epoch"], 2)
            self.assertEqual(
                set(summary["saved_checkpoints"]),
                {"ckpt_step_2", "ckpt_step_3", "end_epoch_1"},
            )
            self.assertTrue((run_directory / "ckpt_step_2").is_dir())
            self.assertTrue((run_directory / "ckpt_step_3").is_dir())
            self.assertTrue((run_directory / "end_epoch_1").is_dir())
            train_logs = [
                metrics for _, metrics in wandb_run.logs if "train/loss" in metrics
            ]
            self.assertEqual(len(train_logs), 1)
            self.assertLess(
                train_logs[0]["train/loss"], train_logs[0]["train/loss_step"]
            )
            self.assertTrue(wandb_run.finished)


if __name__ == "__main__":
    unittest.main()
