"""Extraction of ARC grids from raw Prediction Model completions."""

import re
from dataclasses import dataclass
from typing import Optional

from explain_then_adapt.arc.types import Grid

GRID_LINE = re.compile(r"[0-9]+", flags=re.ASCII)
MODEL_SPECIAL_TOKENS = ("<|im_end|>", "<|im_start|>")


@dataclass(frozen=True)
class GridParseResult:
    """A parsed grid or a stable reason why parsing failed."""

    status: str
    grid: Optional[Grid]

    @property
    def is_valid(self) -> bool:
        return self.status == "ok"


def parse_prediction_grid(
    text: str,
    *,
    max_height: int = 30,
    max_width: int = 30,
) -> GridParseResult:
    """Parse the leading digit-only grid used by the final PM protocol.

    Chat-template markers and an exact leading ``assistant`` line are ignored.
    Text after the first non-grid line is allowed, matching the thesis evaluator.
    """
    if not isinstance(text, str):
        raise TypeError("prediction text must be a string.")
    if isinstance(max_height, bool) or max_height <= 0:
        raise ValueError("max_height must be a positive integer.")
    if isinstance(max_width, bool) or max_width <= 0:
        raise ValueError("max_width must be a positive integer.")

    normalized = text
    for token in MODEL_SPECIAL_TOKENS:
        normalized = normalized.replace(token, "")
    normalized = normalized.strip()
    if not normalized:
        return GridParseResult(status="empty_output", grid=None)

    lines = normalized.splitlines()
    if lines and lines[0].strip().lower() == "assistant":
        lines = lines[1:]

    grid_lines = []
    for line in lines:
        stripped = line.strip()
        if GRID_LINE.fullmatch(stripped):
            grid_lines.append(stripped)
        else:
            break
    if not grid_lines:
        return GridParseResult(status="no_leading_grid", grid=None)
    if len(grid_lines) > max_height:
        return GridParseResult(status="grid_too_tall", grid=None)

    width = len(grid_lines[0])
    if width > max_width:
        return GridParseResult(status="grid_too_wide", grid=None)
    if any(len(row) != width for row in grid_lines):
        return GridParseResult(status="ragged_grid", grid=None)
    return GridParseResult(
        status="ok",
        grid=[[int(character) for character in row] for row in grid_lines],
    )
