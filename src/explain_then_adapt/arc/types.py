"""Shared type definitions for ARC tasks and grids."""

from typing import List, Tuple, TypedDict


Grid = List[List[int]]


class InputExample(TypedDict):
    """An ARC example for which only the input grid is available."""

    input: Grid


class Example(InputExample):
    """An ARC demonstration pair with input and output grids."""

    output: Grid


class Task(TypedDict):
    """The standard train/test structure of an ARC task."""

    train: List[Example]
    test: List[InputExample]


TRANSFORM_CODES: Tuple[str, ...] = (
    "ID",
    "R90",
    "R180",
    "R270",
    "FH",
    "FV",
    "FD1",
    "FD2",
)
