import unittest

from explain_then_adapt.arc.formatting import (
    format_example_to_string,
    format_grid_to_string,
    format_puzzle_to_string,
)


class FormattingTests(unittest.TestCase):
    def test_grid_delimiter_is_configurable(self) -> None:
        grid = [[1, 2], [3, 4]]

        self.assertEqual(format_grid_to_string(grid), "1 2\n3 4")
        self.assertEqual(format_grid_to_string(grid, delimiter=""), "12\n34")

    def test_example_and_puzzle_labels_match_prompt_format(self) -> None:
        example = {"input": [[1]], "output": [[2]]}

        self.assertEqual(
            format_example_to_string(example, index=1),
            "Example 1\nInput:\n1\nOutput:\n2",
        )
        self.assertEqual(
            format_puzzle_to_string([example, example]),
            "Example 1\nInput:\n1\nOutput:\n2\n\n"
            "Example 2\nInput:\n1\nOutput:\n2",
        )

    def test_empty_puzzle_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            format_puzzle_to_string([])


if __name__ == "__main__":
    unittest.main()
