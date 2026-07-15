"""Stable ARC task primitives shared across the project pipeline."""

from .augmented_keys import (
    AugmentedKey,
    apply_order_mapping,
    is_augmented_key,
    make_augmented_key,
    parse_augmented_key,
    parse_order_mapping,
)
from .formatting import (
    format_example_to_string,
    format_examples_to_string,
    format_grid_to_string,
    format_puzzle_to_string,
    load_puzzle_as_string,
)
from .io import (
    load_existing_list,
    load_full_puzzle,
    load_puzzle_test,
    load_puzzle_train,
    load_records,
    load_subset,
    load_task,
)
from .transforms import (
    flip_grid,
    load_and_transform_full_puzzle,
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
from .types import TRANSFORM_CODES, Example, Grid, InputExample, Task


__all__ = [
    "AugmentedKey",
    "Example",
    "Grid",
    "InputExample",
    "TRANSFORM_CODES",
    "Task",
    "apply_order_mapping",
    "flip_grid",
    "format_example_to_string",
    "format_examples_to_string",
    "format_grid_to_string",
    "format_puzzle_to_string",
    "is_augmented_key",
    "load_and_transform_full_puzzle",
    "load_existing_list",
    "load_full_puzzle",
    "load_puzzle_as_string",
    "load_puzzle_test",
    "load_puzzle_train",
    "load_records",
    "load_subset",
    "load_task",
    "make_augmented_key",
    "parse_augmented_key",
    "parse_order_mapping",
    "parse_value_mapping",
    "remap_grid_by_value_mapping",
    "rotate_grid",
    "transform_example",
    "transform_full_puzzle",
    "transform_grid",
    "transform_individual_grid",
    "transform_pairs",
    "transform_puzzle_train",
]
