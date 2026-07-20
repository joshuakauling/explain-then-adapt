import unittest
from pathlib import Path

from explain_then_adapt.evaluation.config import load_evaluation_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "evaluation" / "evaluation.yaml"


class EvaluationConfigTests(unittest.TestCase):
    def test_final_thesis_settings_load(self) -> None:
        config = load_evaluation_config(CONFIG_PATH)

        self.assertEqual(config.grid_parser.max_height, 30)
        self.assertEqual(config.grid_parser.max_width, 30)
        self.assertEqual(config.compute.prefill_tokens_per_second, 5000.0)
        self.assertEqual(config.compute.decode_tokens_per_second, 75.0)
        self.assertEqual(config.compute.training_token_multiplier, 3.0)


if __name__ == "__main__":
    unittest.main()
