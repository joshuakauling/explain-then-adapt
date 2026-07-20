import unittest
from pathlib import Path

from explain_then_adapt.inference.config import (
    GUIDANCE_BUDGETS,
    load_inference_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "inference" / "inference.yaml"


class InferenceConfigTests(unittest.TestCase):
    def test_final_thesis_settings_are_explicit(self) -> None:
        config = load_inference_config(CONFIG_PATH)

        self.assertEqual(config.augmentation_seed, 167)
        self.assertEqual(config.sampling_seed, 42)
        self.assertEqual(config.variants_per_transform, 8)
        self.assertEqual(config.reasoning.engine.max_model_len, 32768)
        self.assertEqual(config.prediction.engine.max_model_len, 8192)
        self.assertEqual(config.prediction.engine.ttt_max_model_len, 16384)
        self.assertEqual(config.prediction.engine.max_lora_rank, 32)
        self.assertEqual(config.protocols.standard_samples, 32)
        self.assertEqual(config.protocols.augmented_variants, 64)
        self.assertEqual(config.protocols.budgeted_total_samples, 64)
        self.assertEqual(config.protocols.guidance_budgets, GUIDANCE_BUDGETS)
        self.assertEqual(
            config.prediction.stop,
            ("<|im_end|>", "<|im_start|>"),
        )


if __name__ == "__main__":
    unittest.main()
