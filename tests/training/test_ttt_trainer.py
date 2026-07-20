import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from explain_then_adapt.training.config import load_ttt_training_config
from explain_then_adapt.training.ttt_trainer import (
    TTT_BOOTSTRAP_ADAPTER,
    build_ttt_scheduler,
    train_ttt_adapters,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "training" / "test_time_training.yaml"

TASK = {
    "train": [
        {"input": [[1, 0]], "output": [[2, 0]]},
        {"input": [[1, 1]], "output": [[2, 2]]},
    ],
    "test": [{"input": [[1]]}],
}


class FakeTokenizer:
    name_or_path = "fake-qwen"
    vocab_size = 256
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"

    def encode(self, text: str, *, add_special_tokens: bool):
        if add_special_tokens:
            raise AssertionError("special tokens must be disabled")
        return list(text.encode("utf-8"))


class TinyTTTModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.active_adapter = TTT_BOOTSTRAP_ADAPTER
        self.added: list[str] = []
        self.deleted: list[str] = []
        self.forward_calls = 0

    def add_adapter(self, task_id: str, _: object) -> None:
        self.anchor = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.active_adapter = task_id
        self.added.append(task_id)

    def set_adapter(self, adapter_name: str) -> None:
        self.active_adapter = adapter_name
        self.anchor.requires_grad_(adapter_name != TTT_BOOTSTRAP_ADAPTER)

    def delete_adapter(self, task_id: str) -> None:
        self.deleted.append(task_id)

    def forward(self, **batch: torch.Tensor) -> SimpleNamespace:
        self.forward_calls += 1
        targets = batch["labels"][batch["labels"] != -100].float()
        target = targets.mean() / 100.0
        return SimpleNamespace(loss=(self.anchor - target).square())

    def save_pretrained(
        self,
        directory: Path,
        *,
        selected_adapters,
        safe_serialization: bool,
    ) -> None:
        if not safe_serialization or selected_adapters != [self.active_adapter]:
            raise AssertionError("only the active task adapter may be saved")
        nested = directory / self.active_adapter
        nested.mkdir(parents=True)
        with (nested / "adapter_config.json").open("w", encoding="utf-8") as file:
            json.dump({"adapter": self.active_adapter}, file)
        (nested / "adapter_model.safetensors").write_bytes(b"test-adapter")


def _config():
    base = load_ttt_training_config(
        CONFIG_PATH,
        "guided",
        guidance_budget=0,
    )
    return replace(
        base,
        model=replace(
            base.model,
            dtype="float32",
            attention_implementation="sdpa",
            gradient_checkpointing=False,
        ),
        data=replace(base.data, pad_to_multiple_of=1),
        loader=replace(base.loader, pin_memory=False),
    )


class TTTSchedulerTests(unittest.TestCase):
    def test_schedule_reaches_warmup_peak_and_cosine_floor(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.SGD([parameter], lr=1.25e-4)
        scheduler = build_ttt_scheduler(
            optimizer,
            total_steps=64,
            warmup_ratio=0.5,
            warmup_start_learning_rate=0.0,
            peak_learning_rate=1.25e-4,
            min_learning_rate=5.0e-6,
        )
        learning_rates = [optimizer.param_groups[0]["lr"]]
        for _ in range(64):
            optimizer.step()
            scheduler.step()
            learning_rates.append(optimizer.param_groups[0]["lr"])

        self.assertAlmostEqual(learning_rates[0], 0.0)
        self.assertAlmostEqual(learning_rates[32], 1.25e-4)
        self.assertAlmostEqual(learning_rates[64], 5.0e-6)


class TTTTrainingLoopTests(unittest.TestCase):
    def test_trains_fresh_64_step_adapters_and_resumes_complete_tasks(self) -> None:
        config = _config()
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            tasks_directory = directory / "tasks"
            tasks_directory.mkdir()
            for task_id in ("abc123", "def456"):
                with (tasks_directory / f"{task_id}.json").open(
                    "w", encoding="utf-8"
                ) as file:
                    json.dump(TASK, file)
            prediction_model = directory / "prediction-model"
            prediction_model.mkdir()
            model = TinyTTTModel()
            model_builds = []

            def build_model(*args, **kwargs):
                model_builds.append((args, kwargs))
                return model, object()

            def build_optimizer(candidate_model, candidate_config):
                return torch.optim.SGD(
                    [
                        parameter
                        for parameter in candidate_model.parameters()
                        if parameter.requires_grad
                    ],
                    lr=candidate_config.profile.peak_learning_rate,
                )

            with (
                patch(
                    "explain_then_adapt.training.ttt_trainer.load_ttt_tokenizer",
                    return_value=FakeTokenizer(),
                ),
                patch(
                    "explain_then_adapt.training.ttt_trainer.build_ttt_model",
                    side_effect=build_model,
                ),
                patch(
                    "explain_then_adapt.training.ttt_trainer.build_ttt_optimizer",
                    side_effect=build_optimizer,
                ),
                patch(
                    "explain_then_adapt.training.ttt_trainer.tqdm",
                    side_effect=lambda iterable, **_: iterable,
                ),
            ):
                summary = train_ttt_adapters(
                    config=config,
                    tasks_directory=tasks_directory,
                    task_ids=["def456", "abc123"],
                    prediction_model_path=prediction_model,
                    output_directory=directory / "runs",
                    run_name="guided-budget-zero",
                )
                resumed = train_ttt_adapters(
                    config=config,
                    tasks_directory=tasks_directory,
                    task_ids=["def456.json", "abc123.json"],
                    prediction_model_path=prediction_model,
                    output_directory=directory / "runs",
                    run_name="guided-budget-zero",
                    resume=True,
                )

            self.assertEqual(summary, resumed)
            self.assertEqual(summary["completed_task_count"], 2)
            self.assertEqual(summary["optimizer_updates"], 128)
            self.assertEqual(model.forward_calls, 128)
            self.assertEqual(model.added, ["abc123", "def456"])
            self.assertEqual(model.deleted, ["abc123", "def456"])
            self.assertEqual(len(model_builds), 1)
            for task_id in ("abc123", "def456"):
                task_directory = directory / "runs" / "guided-budget-zero" / task_id
                self.assertTrue(
                    (task_directory / "adapter_model.safetensors").is_file()
                )
                with (task_directory / "ttt_manifest.json").open(
                    "r", encoding="utf-8"
                ) as file:
                    manifest = json.load(file)
                self.assertEqual(manifest["optimizer_updates"], 64)
                self.assertEqual(manifest["guidance_counts"], {"budget_empty": 64})
                self.assertEqual(
                    manifest["transform_counts"],
                    {
                        "FD1": 8,
                        "FD2": 8,
                        "FH": 8,
                        "FV": 8,
                        "ID": 8,
                        "R180": 8,
                        "R270": 8,
                        "R90": 8,
                    },
                )

            manifest_path = (
                directory
                / "runs"
                / "guided-budget-zero"
                / "abc123"
                / "ttt_manifest.json"
            )
            with manifest_path.open("r", encoding="utf-8") as file:
                incomplete_manifest = json.load(file)
            incomplete_manifest["optimizer_updates"] = 63
            with manifest_path.open("w", encoding="utf-8") as file:
                json.dump(incomplete_manifest, file)
            with (
                patch(
                    "explain_then_adapt.training.ttt_trainer.load_ttt_tokenizer",
                    return_value=FakeTokenizer(),
                ),
                self.assertRaisesRegex(RuntimeError, "incomplete"),
            ):
                train_ttt_adapters(
                    config=config,
                    tasks_directory=tasks_directory,
                    task_ids=["def456", "abc123"],
                    prediction_model_path=prediction_model,
                    output_directory=directory / "runs",
                    run_name="guided-budget-zero",
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
