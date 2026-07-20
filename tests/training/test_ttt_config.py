import unittest
from pathlib import Path

from explain_then_adapt.training.config import load_ttt_training_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "training" / "test_time_training.yaml"


class TTTTrainingConfigTests(unittest.TestCase):
    def test_final_guided_and_unguided_profiles(self) -> None:
        guided = load_ttt_training_config(CONFIG_PATH, "guided")
        unguided = load_ttt_training_config(CONFIG_PATH, "unguided")

        self.assertEqual(guided.model.name, "Qwen/Qwen3-4B-Thinking-2507")
        self.assertEqual(guided.model.quantization_bits, 4)
        self.assertEqual(guided.lora.rank, 32)
        self.assertEqual(guided.lora.alpha, 16)
        self.assertEqual(guided.lora.dropout, 0.0)
        self.assertFalse(guided.lora.use_rslora)
        self.assertEqual(guided.data.variants_per_task, 64)
        self.assertTrue(guided.data.ignore_first_response)
        self.assertEqual(guided.data.empty_guidance_content, " ")
        self.assertEqual(guided.optimization.warmup_ratio, 0.5)
        self.assertEqual(guided.optimization.min_learning_rate, 5.0e-6)
        self.assertEqual(guided.profile.guidance_budget, 64)
        self.assertEqual(guided.profile.peak_learning_rate, 1.25e-4)
        self.assertEqual(unguided.profile.guidance_budget, 0)
        self.assertEqual(unguided.profile.peak_learning_rate, 2.0e-4)

    def test_guided_budget_and_historical_missing_policy_are_explicit(self) -> None:
        config = load_ttt_training_config(
            CONFIG_PATH,
            "guided",
            guidance_budget=16,
            missing_guidance_policy="omit_system",
        )

        self.assertEqual(config.profile.guidance_budget, 16)
        self.assertEqual(config.data.missing_guidance_policy, "omit_system")

    def test_invalid_budget_and_unguided_override_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "guidance_budget"):
            load_ttt_training_config(
                CONFIG_PATH,
                "guided",
                guidance_budget=24,
            )
        with self.assertRaisesRegex(ValueError, "unguided TTT"):
            load_ttt_training_config(
                CONFIG_PATH,
                "unguided",
                guidance_budget=8,
            )


if __name__ == "__main__":
    unittest.main()
