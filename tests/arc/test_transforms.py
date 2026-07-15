import json
import tempfile
import unittest
from pathlib import Path

from explain_then_adapt.arc.transforms import (
    flip_grid,
    parse_value_mapping,
    remap_grid_by_value_mapping,
    rotate_grid,
    transform_example,
    transform_full_puzzle,
    transform_grid,
    transform_individual_grid,
    transform_pairs,
    transform_puzzle_train,
)


class TransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = [[1, 2, 3], [4, 5, 6]]

    def test_rectangular_grid_rotations(self) -> None:
        self.assertEqual(rotate_grid(self.grid, 1), [[4, 1], [5, 2], [6, 3]])
        self.assertEqual(rotate_grid(self.grid, 2), [[6, 5, 4], [3, 2, 1]])
        self.assertEqual(self.grid, [[1, 2, 3], [4, 5, 6]])

    def test_grid_reflections(self) -> None:
        self.assertEqual(flip_grid(self.grid, "horizontal"), [[4, 5, 6], [1, 2, 3]])
        self.assertEqual(flip_grid(self.grid, "vertical"), [[3, 2, 1], [6, 5, 4]])
        self.assertEqual(flip_grid(self.grid, "main_diagonal"), [[1, 4], [2, 5], [3, 6]])
        self.assertEqual(flip_grid(self.grid, "anti_diagonal"), [[6, 3], [5, 2], [4, 1]])

    def test_transform_codes_are_case_insensitive(self) -> None:
        self.assertEqual(transform_grid(self.grid, "r90"), [[4, 1], [5, 2], [6, 3]])

    def test_value_mapping_is_validated_and_applied(self) -> None:
        mapping = "0213456789"

        self.assertEqual(parse_value_mapping(mapping), [0, 2, 1, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(remap_grid_by_value_mapping([[0, 1, 2]], mapping), [[0, 2, 1]])
        with self.assertRaises(ValueError):
            parse_value_mapping("0012345678")

    def test_legacy_transformation_keywords_remain_supported(self) -> None:
        transformed = transform_individual_grid(
            grid=[[1, 2]],
            transformation_code="R90",
            value_mapping_str="0123456789",
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {
                "train": [{"input": [[1]], "output": [[2]]}],
                "test": [{"input": [[3]]}],
            }
            (root / "abc123.json").write_text(json.dumps(task), encoding="utf-8")
            puzzle = transform_puzzle_train(
                orig_key="abc123",
                puzzle_path=root,
                transformation_label="ID",
                value_mapping_str="0123456789",
                reorder_mapping="0",
            )

        self.assertEqual(transformed, [[1], [2]])
        self.assertEqual(puzzle, task["train"])

    def test_example_transformation_and_reordering(self) -> None:
        first = {"input": [[1, 2]], "output": [[2, 1]]}
        second = {"input": [[3, 4]], "output": [[4, 3]]}
        identity_mapping = "0123456789"

        transformed = transform_example(first, "R90", identity_mapping)
        reordered = transform_pairs(
            [first, second],
            "ID",
            identity_mapping,
            "10",
        )

        self.assertEqual(transformed, {"input": [[1], [2]], "output": [[2], [1]]})
        self.assertEqual(reordered, [second, first])
        self.assertEqual(first, {"input": [[1, 2]], "output": [[2, 1]]})

    def test_full_puzzle_requires_labelled_test_pairs(self) -> None:
        with self.assertRaises(ValueError):
            transform_full_puzzle(
                [{"input": [[1]]}],
                "ID",
                "0123456789",
            )


if __name__ == "__main__":
    unittest.main()
