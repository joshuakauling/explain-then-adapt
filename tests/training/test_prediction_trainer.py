import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import torch

from explain_then_adapt.training.config import (
    PredictionTrainingConfig,
    load_prediction_training_config,
)
from explain_then_adapt.training.prediction_data import prediction_cache_contract
from explain_then_adapt.training.prediction_trainer import (
    PredictionAssistantCollator,
    PredictionTaskVariantDataset,
    train_prediction_model,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "training" / "prediction_model.yaml"


def _training_config() -> PredictionTrainingConfig:
    base = load_prediction_training_config(CONFIG_PATH, "guided")
    return replace(
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
            synthetic_task_count=2,
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
            checkpoint_steps=(4,),
        ),
        loader=replace(base.loader, pin_memory=False),
    )


class PredictionDatasetAndCollatorTests(unittest.TestCase):
    def test_dataset_uses_one_variant_per_task_and_epoch(self) -> None:
        dataset = PredictionTaskVariantDataset(
            {
                "a": [
                    {"input_ids": [1, 2], "assistant_spans": [[1, 2]], "id": 0},
                    {"input_ids": [1, 3], "assistant_spans": [[1, 2]], "id": 1},
                ],
                "b": [
                    {"input_ids": [4, 5], "assistant_spans": [[1, 2]], "id": 0},
                    {"input_ids": [4, 6], "assistant_spans": [[1, 2]], "id": 1},
                ],
            },
            seed=167,
        )

        dataset.set_epoch(1)
        self.assertEqual({dataset[index]["id"] for index in range(2)}, {1})
        with self.assertRaisesRegex(ValueError, "requires at least"):
            dataset.set_epoch(2)

    def test_collator_masks_only_selected_grid_spans(self) -> None:
        feature = {
            "input_ids": [10, 11, 12, 13, 14, 15, 16, 17, 18],
            "assistant_spans": [[2, 4], [6, 8]],
        }
        masked = PredictionAssistantCollator(
            pad_token_id=0,
            ignore_first_response=True,
            pad_to_multiple_of=4,
        )([feature])
        self.assertEqual(masked["input_ids"].shape, (1, 12))
        self.assertEqual(
            masked["labels"].tolist()[0],
            [-100, -100, -100, -100, -100, -100, 16, 17, -100, -100, -100, -100],
        )

        see_first = PredictionAssistantCollator(
            pad_token_id=0,
            ignore_first_response=False,
            pad_to_multiple_of=1,
        )([feature])
        self.assertEqual(see_first["labels"][0, 2:4].tolist(), [12, 13])
        self.assertEqual(see_first["labels"][0, 6:8].tolist(), [16, 17])

    def test_masking_the_only_response_is_rejected(self) -> None:
        collator = PredictionAssistantCollator(
            pad_token_id=0,
            ignore_first_response=True,
            pad_to_multiple_of=1,
        )
        with self.assertRaisesRegex(ValueError, "no target"):
            collator([{"input_ids": [1, 2], "assistant_spans": [[1, 2]]}])


class TinyPredictionModel(torch.nn.Module):
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
        self.logs: List[Tuple[int, Dict[str, Any]]] = []
        self.finished = False

    def log(self, metrics, *, step: int) -> None:
        self.logs.append((step, dict(metrics)))

    def finish(self) -> None:
        self.finished = True


class PredictionTrainingLoopTests(unittest.TestCase):
    def test_loop_preserves_profile_masking_and_candidate_checkpoints(self) -> None:
        config = _training_config()

        def record(task_id: str, variant_id: str, first: int, second: int):
            return {
                "task_id": task_id,
                "variant_id": variant_id,
                "input_ids": torch.tensor(
                    [90, first, 91, second],
                    dtype=torch.int32,
                ),
                "assistant_spans": [[1, 2], [3, 4]],
            }

        cache = {
            "schema_version": 1,
            "kind": "prediction_model",
            "metadata": {
                "cache_contract": prediction_cache_contract(config),
                "tokenizer": {"pad_token_id": 0},
            },
            "splits": {
                "train": {
                    "task-a": [
                        record("task-a", "a-0", 100, 1),
                        record("task-a", "a-1", 100, 5),
                    ],
                    "task-b": [
                        record("task-b", "b-0", 100, 3),
                        record("task-b", "b-1", 100, 7),
                    ],
                },
                "validation": {
                    "val": [record("val", "val", 100, 2)],
                },
                "validation_augmented": {
                    "val": [record("val", "val-aug", 100, 4)],
                },
            },
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            cache_path = directory / "prediction.pt"
            torch.save(cache, cache_path)
            model = TinyPredictionModel()
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.optimization.peak_learning_rate,
            )
            wandb_run = FakeWandbRun()
            wandb_module = types.ModuleType("wandb")
            wandb_module.init = lambda **_: wandb_run  # type: ignore[attr-defined]

            with (
                patch(
                    "explain_then_adapt.training.prediction_trainer."
                    "build_prediction_model_and_optimizer",
                    return_value=(model, optimizer),
                ),
                patch.dict("sys.modules", {"wandb": wandb_module}),
            ):
                summary = train_prediction_model(
                    config=config,
                    cache_path=cache_path,
                    output_directory=directory / "runs",
                    run_name="test-run",
                    use_wandb=True,
                )

            self.assertEqual(summary["profile"], "guided")
            self.assertEqual(summary["global_steps"], 4)
            self.assertEqual(
                set(summary["saved_checkpoints"]),
                {"ckpt_step_2", "ckpt_step_3", "ckpt_step_4", "end_epoch_1"},
            )
            train_logs = [
                metrics for _, metrics in wandb_run.logs if "train/loss" in metrics
            ]
            self.assertEqual(len(train_logs), 1)
            self.assertLess(train_logs[0]["train/loss"], 10)
            self.assertTrue(wandb_run.finished)

    def test_guided_rearc_requires_an_initial_merged_model(self) -> None:
        base = load_prediction_training_config(CONFIG_PATH, "guided_rearc")
        with self.assertRaisesRegex(ValueError, "requires --initial-model"):
            train_prediction_model(
                config=base,
                cache_path=Path("unused.pt"),
                output_directory=Path("unused"),
                run_name="guided-rearc",
                use_wandb=False,
            )


if __name__ == "__main__":
    unittest.main()
