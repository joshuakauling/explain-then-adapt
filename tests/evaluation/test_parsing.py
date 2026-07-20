import unittest

from explain_then_adapt.evaluation.parsing import parse_prediction_grid


class PredictionGridParsingTests(unittest.TestCase):
    def test_parses_chat_artifacts_and_ignores_trailing_text(self) -> None:
        result = parse_prediction_grid(
            "<|im_start|>assistant\n012\n345\nexplanation<|im_end|>"
        )

        self.assertTrue(result.is_valid)
        self.assertEqual(result.grid, [[0, 1, 2], [3, 4, 5]])

    def test_rejects_empty_non_grid_and_ragged_outputs(self) -> None:
        self.assertEqual(parse_prediction_grid("  ").status, "empty_output")
        self.assertEqual(
            parse_prediction_grid("```text\n12\n34").status,
            "no_leading_grid",
        )
        self.assertEqual(parse_prediction_grid("12\n3").status, "ragged_grid")

    def test_enforces_arc_dimensions(self) -> None:
        self.assertEqual(
            parse_prediction_grid("123", max_width=2).status,
            "grid_too_wide",
        )
        self.assertEqual(
            parse_prediction_grid("1\n2", max_height=1).status,
            "grid_too_tall",
        )


if __name__ == "__main__":
    unittest.main()
