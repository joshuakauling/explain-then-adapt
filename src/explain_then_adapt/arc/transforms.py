"""Geometric and value transformations for ARC grids and examples."""

from typing import List, Optional, cast

from .augmented_keys import apply_order_mapping
from .io import PathLike, load_full_puzzle, load_puzzle_train
from .types import TRANSFORM_CODES, Example, Grid, InputExample


def _copy_and_validate_grid(grid: Grid) -> Grid:
    if not isinstance(grid, list) or not grid or any(not isinstance(row, list) or not row for row in grid):
        raise ValueError("grid must be a non-empty list of non-empty rows.")

    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("grid rows must all have the same length.")
    return [row[:] for row in grid]


def rotate_grid(grid: Grid, k: int = 0) -> Grid:
    """Rotate a grid clockwise by 90 degrees ``k`` times."""
    rotated = _copy_and_validate_grid(grid)
    for _ in range(k % 4):
        rotated = [list(row) for row in zip(*reversed(rotated))]
    return rotated


def flip_grid(grid: Grid, mode: str = "horizontal") -> Grid:
    """Reflect a grid horizontally, vertically, or across either diagonal."""
    copied = _copy_and_validate_grid(grid)
    if mode == "horizontal":
        return list(reversed(copied))
    if mode == "vertical":
        return [list(reversed(row)) for row in copied]

    transposed = [list(row) for row in zip(*copied)]
    if mode == "main_diagonal":
        return transposed
    if mode == "anti_diagonal":
        return [list(reversed(row)) for row in reversed(transposed)]
    raise ValueError(
        "mode must be one of: horizontal, vertical, main_diagonal, anti_diagonal."
    )


def transform_grid(grid: Grid, code: str) -> Grid:
    """Apply one of the eight square-symmetry transformations to a grid."""
    normalized_code = code.upper()
    operations = {
        "ID": lambda value: _copy_and_validate_grid(value),
        "R90": lambda value: rotate_grid(value, 1),
        "R180": lambda value: rotate_grid(value, 2),
        "R270": lambda value: rotate_grid(value, 3),
        "FH": lambda value: flip_grid(value, "horizontal"),
        "FV": lambda value: flip_grid(value, "vertical"),
        "FD1": lambda value: flip_grid(value, "main_diagonal"),
        "FD2": lambda value: flip_grid(value, "anti_diagonal"),
    }
    try:
        return operations[normalized_code](grid)
    except KeyError as error:
        raise ValueError(
            f"unknown transformation code {code!r}; expected one of {TRANSFORM_CODES}."
        ) from error


def parse_value_mapping(mapping: str) -> List[int]:
    """Parse a permutation in which position ``i`` gives the new value for ``i``."""
    if not isinstance(mapping, str):
        raise TypeError("mapping must be a string.")
    if len(mapping) != 10 or not mapping.isascii() or not mapping.isdigit():
        raise ValueError("mapping must contain exactly 10 ASCII digits.")
    if set(mapping) != set("0123456789"):
        raise ValueError("mapping must be a permutation of digits 0..9.")
    return [int(character) for character in mapping]


def remap_grid_by_value_mapping(grid: Grid, mapping: str) -> Grid:
    """Apply a value permutation to every cell in a grid."""
    copied = _copy_and_validate_grid(grid)
    parsed_mapping = parse_value_mapping(mapping)

    for row in copied:
        for value in row:
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 9:
                raise ValueError("grid values must be integers in range 0..9.")
    return [[parsed_mapping[value] for value in row] for row in copied]


def transform_individual_grid(
    grid: Grid,
    transformation_code: str,
    value_mapping_str: str,
) -> Grid:
    """Apply value remapping followed by a geometric transformation."""
    remapped = remap_grid_by_value_mapping(grid, value_mapping_str)
    return transform_grid(remapped, transformation_code)


def transform_example(
    example: Example,
    transformation_code: str,
    value_mapping: str,
) -> Example:
    """Transform both grids of one ARC demonstration pair."""
    if "input" not in example or "output" not in example:
        raise ValueError("example must contain 'input' and 'output' grids.")
    return {
        "input": transform_individual_grid(
            example["input"], transformation_code, value_mapping
        ),
        "output": transform_individual_grid(
            example["output"], transformation_code, value_mapping
        ),
    }


def transform_pairs(
    pairs: List[Example],
    transformation_code: str,
    value_mapping: str,
    order_mapping: Optional[str] = None,
) -> List[Example]:
    """Reorder and transform a list of ARC demonstration pairs."""
    ordered_pairs = apply_order_mapping(pairs, order_mapping)
    return [
        transform_example(pair, transformation_code, value_mapping)
        for pair in ordered_pairs
    ]


def transform_puzzle_train(
    orig_key: str,
    puzzle_path: PathLike,
    transformation_label: str,
    value_mapping_str: str,
    reorder_mapping: str,
) -> List[Example]:
    """Load, reorder, and transform a task's demonstration pairs."""
    return transform_pairs(
        load_puzzle_train(orig_key, puzzle_path),
        transformation_label,
        value_mapping_str,
        reorder_mapping,
    )


def transform_full_puzzle(
    puzzle: List[InputExample],
    transformation_label: str,
    value_mapping_str: str,
) -> List[Example]:
    """Transform train and labelled test pairs represented as one list."""
    labelled_pairs: List[Example] = []
    for pair in puzzle:
        if "output" not in pair:
            raise ValueError("full-puzzle transformation requires labelled test pairs.")
        labelled_pairs.append(cast(Example, pair))
    return transform_pairs(labelled_pairs, transformation_label, value_mapping_str)


def load_and_transform_full_puzzle(
    puzzle_id: str,
    puzzle_path: PathLike,
    transformation_label: str,
    value_mapping_str: str,
) -> List[Example]:
    """Load and transform all labelled pairs of one ARC task."""
    return transform_full_puzzle(
        load_full_puzzle(puzzle_id, puzzle_path),
        transformation_label,
        value_mapping_str,
    )
