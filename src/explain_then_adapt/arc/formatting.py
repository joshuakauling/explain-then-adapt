"""Text formatting helpers for ARC grids and examples."""

from typing import List, Optional

from .io import PathLike, load_puzzle_train
from .types import Example, Grid


def format_grid_to_string(grid: Grid, delimiter: str = " ") -> str:
    """Render a grid as rows separated by newlines."""
    if not isinstance(grid, list) or any(not isinstance(row, list) for row in grid):
        raise ValueError("grid must be a list of lists.")
    return "\n".join(delimiter.join(str(cell) for cell in row) for row in grid)


def format_example_to_string(
    example: Example,
    index: Optional[int] = None,
    delimiter: str = " ",
) -> str:
    """Render one input-output demonstration as a labelled text block."""
    if not isinstance(example, dict) or "input" not in example or "output" not in example:
        raise ValueError("example must contain 'input' and 'output' grids.")

    lines: List[str] = []
    if index is not None:
        lines.append(f"Example {index}")
    lines.extend(
        (
            "Input:",
            format_grid_to_string(example["input"], delimiter),
            "Output:",
            format_grid_to_string(example["output"], delimiter),
        )
    )
    return "\n".join(lines)


def format_examples_to_string(examples: List[Example], delimiter: str = " ") -> str:
    """Render multiple demonstrations with one blank line between examples."""
    if not isinstance(examples, list) or not examples:
        raise ValueError("examples must be a non-empty list.")
    return "\n\n".join(
        format_example_to_string(example, index + 1, delimiter)
        for index, example in enumerate(examples)
    )


def format_puzzle_to_string(puzzle: List[Example], delimiter: str = " ") -> str:
    """Render an ARC puzzle's demonstration pairs as text."""
    return format_examples_to_string(puzzle, delimiter)


def load_puzzle_as_string(
    puzzle_id: str,
    puzzle_path: PathLike,
    delimiter: str = " ",
) -> str:
    """Load and render the demonstration pairs of one ARC task."""
    return format_puzzle_to_string(
        load_puzzle_train(puzzle_id, puzzle_path),
        delimiter,
    )
