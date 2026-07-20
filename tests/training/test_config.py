import unittest
from dataclasses import replace
from pathlib import Path

import torch
from torch.optim.lr_scheduler import SequentialLR

from explain_then_adapt.training.config import (
    LoaderSettings,
    load_reasoning_training_config,
)
from explain_then_adapt.training.reasoning_trainer import (
    build_cosine_scheduler,
    optimizer_steps_per_epoch,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "training" / "reasoning_model.yaml"


class ReasoningTrainingConfigTests(unittest.TestCase):
    def test_final_thesis_configuration(self) -> None:
        config = load_reasoning_training_config(CONFIG_PATH)

        self.assertEqual(config.model.name, "Qwen/Qwen3-4B-Thinking-2507")
        self.assertEqual(config.model.quantization_bits, 4)
        self.assertEqual(config.lora.rank, 128)
        self.assertEqual(config.lora.alpha, 32)
        self.assertEqual(config.data.variants_per_task, 100)
        self.assertEqual(config.data.max_sequence_length, 8192)
        self.assertEqual(config.optimization.epochs, 100)
        self.assertEqual(config.optimization.micro_batch_size, 2)
        self.assertEqual(config.optimization.gradient_accumulation_steps, 8)
        self.assertEqual(config.optimization.peak_learning_rate, 1.0e-4)
        self.assertEqual(config.optimization.end_learning_rate, 2.0e-5)
        self.assertEqual(config.optimization.warmup_ratio, 0.05)
        self.assertEqual(config.control.validate_every_steps, 80)
        self.assertEqual(config.control.checkpoint_every_steps, 400)
        self.assertEqual(config.control.checkpoint_epochs, (20, 30))

        steps_per_epoch = optimizer_steps_per_epoch(
            624,
            config.optimization.micro_batch_size,
            config.optimization.gradient_accumulation_steps,
        )
        self.assertEqual(steps_per_epoch, 39)
        self.assertEqual(20 * steps_per_epoch, 780)
        self.assertEqual(30 * steps_per_epoch, 1170)
        self.assertEqual(100 * steps_per_epoch, 3900)

    def test_variant_count_is_configurable_but_bounds_epochs(self) -> None:
        config = load_reasoning_training_config(CONFIG_PATH)
        reduced = replace(
            config,
            data=replace(config.data, variants_per_task=64),
            optimization=replace(config.optimization, epochs=64),
            control=replace(config.control, checkpoint_epochs=(64,)),
        )

        self.assertEqual(reduced.data.variants_per_task, 64)
        self.assertEqual(reduced.optimization.epochs, 64)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            replace(reduced, optimization=replace(reduced.optimization, epochs=65))

    def test_persistent_workers_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be false"):
            LoaderSettings(
                num_workers=2,
                pin_memory=True,
                persistent_workers=True,
            )

    def test_cosine_schedule_reaches_configured_endpoints(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0e-4)
        scheduler = build_cosine_scheduler(
            optimizer,
            total_steps=100,
            warmup_ratio=0.1,
            warmup_start_factor=0.1,
            peak_learning_rate=1.0e-4,
            end_learning_rate=2.0e-5,
        )
        learning_rates = [optimizer.param_groups[0]["lr"]]
        for _ in range(100):
            optimizer.step()
            scheduler.step()
            learning_rates.append(optimizer.param_groups[0]["lr"])

        self.assertIsInstance(scheduler, SequentialLR)
        self.assertAlmostEqual(learning_rates[0], 1.0e-5)
        self.assertAlmostEqual(learning_rates[10], 1.0e-4)
        self.assertAlmostEqual(learning_rates[100], 2.0e-5)


if __name__ == "__main__":
    unittest.main()
