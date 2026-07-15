"""File loading helpers for ARC tasks and related JSON records."""

import json
from pathlib import Path
from typing import Any, List, Union, cast

from .types import Example, InputExample, Task


PathLike = Union[str, Path]


def _json_path(identifier: str, directory: PathLike) -> Path:
    filename = identifier if identifier.endswith(".json") else f"{identifier}.json"
    return Path(directory) / filename


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_subset(subset_name: str, subset_path: PathLike) -> List[str]:
    """Load a JSON list of task filenames and return their task identifiers."""
    subset = _load_json(_json_path(subset_name, subset_path))
    if not isinstance(subset, list) or not all(isinstance(key, str) for key in subset):
        raise ValueError("subset JSON must contain a list of task identifiers.")
    return [key.split(".", maxsplit=1)[0] for key in subset]


def load_task(puzzle_id: str, puzzle_path: PathLike) -> Task:
    """Load one ARC task, including its train and test sections."""
    task = _load_json(_json_path(puzzle_id, puzzle_path))
    if not isinstance(task, dict) or not isinstance(task.get("train"), list) or not isinstance(task.get("test"), list):
        raise ValueError("ARC task JSON must contain list-valued 'train' and 'test' fields.")
    return cast(Task, task)


def load_puzzle_train(puzzle_id: str, puzzle_path: PathLike) -> List[Example]:
    """Load the demonstration pairs of one ARC task."""
    return load_task(puzzle_id, puzzle_path)["train"]


def load_puzzle_test(puzzle_id: str, puzzle_path: PathLike) -> List[InputExample]:
    """Load the test examples of one ARC task."""
    return load_task(puzzle_id, puzzle_path)["test"]


def load_full_puzzle(puzzle_id: str, puzzle_path: PathLike) -> List[InputExample]:
    """Load the train and test examples of one ARC task as a single list."""
    task = load_task(puzzle_id, puzzle_path)
    return [*task["train"], *task["test"]]


def load_records(key: str, records_path: PathLike) -> Any:
    """Load a JSON record identified by its filename stem."""
    return _load_json(_json_path(key, records_path))


def load_existing_list(path: PathLike) -> List[Any]:
    """Load a JSON list if the file exists; otherwise return an empty list."""
    file_path = Path(path)
    if not file_path.exists():
        return []

    value = _load_json(file_path)
    if not isinstance(value, list):
        raise ValueError("existing JSON file must contain a list.")
    return value
