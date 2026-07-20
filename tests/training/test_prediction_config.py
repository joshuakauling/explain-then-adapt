import unittest
from pathlib import Path

from explain_then_adapt.training.config import load_prediction_training_config
from explain_then_adapt.training.reasoning_trainer import optimizer_steps_per_epoch

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "training" / "prediction_model.yaml"


class PredictionTrainingConfigTests(unittest.TestCase):
    def test_final_profiles_resolve_the_thesis_runs(self) -> None:
        guided = load_prediction_training_config(CONFIG_PATH, "guided")
        see_first = load_prediction_training_config(
            CONFIG_PATH,
            "guided_see_first",
        )
        unguided = load_prediction_training_config(CONFIG_PATH, "unguided")
        unguided_rearc = load_prediction_training_config(
            CONFIG_PATH,
            "unguided_rearc",
        )
        guided_rearc = load_prediction_training_config(
            CONFIG_PATH,
            "guided_rearc",
        )

        self.assertEqual(guided.model.name, "Qwen/Qwen3-4B-Thinking-2507")
        self.assertEqual(guided.lora.rank, 128)
        self.assertEqual(guided.data.variants_per_task, 100)
        self.assertEqual(guided.data.max_sequence_length, 8192)
        self.assertTrue(guided.profile.guided)
        self.assertTrue(guided.profile.ignore_first_response)
        self.assertEqual(guided.control.checkpoint_steps, (2500,))

        synthetic_steps = optimizer_steps_per_epoch(624, 2, 8)
        self.assertEqual(synthetic_steps, 39)
        self.assertEqual(synthetic_steps * 40, 1560)
        self.assertEqual(synthetic_steps * 60, 2340)
        self.assertEqual(synthetic_steps * 100, 3900)
        self.assertFalse(see_first.profile.ignore_first_response)
        self.assertEqual(see_first.control.checkpoint_epochs, (40,))
        self.assertFalse(unguided.profile.guided)
        self.assertEqual(unguided.control.checkpoint_epochs, (40, 60))

        rearc_steps = optimizer_steps_per_epoch(400, 2, 8)
        self.assertEqual(rearc_steps, 25)
        self.assertEqual(unguided_rearc.profile.data_source, "rearc")
        self.assertEqual(rearc_steps * unguided_rearc.optimization.epochs, 2500)
        self.assertEqual(guided_rearc.profile.initialization, "merged_model")
        self.assertEqual(guided_rearc.optimization.epochs, 40)

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown Prediction Model profile"):
            load_prediction_training_config(CONFIG_PATH, "missing")


if __name__ == "__main__":
    unittest.main()
