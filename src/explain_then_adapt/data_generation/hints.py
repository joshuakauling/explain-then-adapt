"""Loading and formatting of optional manually curated ARC hints."""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from .records import read_jsonl


PathLike = Union[str, Path]
HINT_FIELDS: Tuple[str, ...] = (
    "general",
    "inputs",
    "outputs",
    "transformation",
    "transformation_steps",
)


class HintStatus(str, Enum):
    """Completeness of a task's optional hint file."""

    COMPLETE = "complete"
    MISSING = "missing"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class Hint:
    """The five hint fields used by the original generation pipeline."""

    general: str
    inputs: str
    outputs: str
    transformation: str
    transformation_steps: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Hint":
        normalized: Dict[str, str] = {}
        for field_name in HINT_FIELDS:
            field_value = value.get(field_name)
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(
                    f"hint field {field_name!r} must be a non-empty string."
                )
            normalized[field_name] = field_value.strip()
        return cls(**normalized)

    def to_dict(self) -> Dict[str, str]:
        return {field_name: getattr(self, field_name) for field_name in HINT_FIELDS}

    def format(self) -> str:
        """Render the hint in the prompt format used for initial generation."""
        labels = (
            ("General", self.general),
            ("Inputs", self.inputs),
            ("Outputs", self.outputs),
            ("Transformation", self.transformation),
            ("Transformation Steps", self.transformation_steps),
        )
        return "\n".join(
            line for label, value in labels for line in (f"{label}:", value)
        )


@dataclass(frozen=True)
class HintLoadResult:
    """A hint plus explicit missing or incomplete provenance."""

    status: HintStatus
    hint: Optional[Hint] = None
    missing_fields: Tuple[str, ...] = ()


def _hint_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, list):
        if not value or not isinstance(value[0], Mapping):
            raise ValueError(
                "legacy hint JSON must contain an object as its first item."
            )
        return value[0]
    if isinstance(value, Mapping):
        return value
    raise ValueError("hint JSON must contain an object or a list containing an object.")


def load_hint_file(path: PathLike) -> HintLoadResult:
    """Load one hint file without silently accepting partial labels."""
    hint_path = Path(path)
    if not hint_path.exists():
        return HintLoadResult(status=HintStatus.MISSING)

    with hint_path.open("r", encoding="utf-8") as file:
        value = _hint_mapping(json.load(file))

    missing_fields = tuple(
        field_name
        for field_name in HINT_FIELDS
        if not isinstance(value.get(field_name), str) or not value[field_name].strip()
    )
    if missing_fields:
        return HintLoadResult(
            status=HintStatus.INCOMPLETE,
            missing_fields=missing_fields,
        )
    return HintLoadResult(status=HintStatus.COMPLETE, hint=Hint.from_mapping(value))


def load_task_hint(task_id: str, hints_directory: PathLike) -> HintLoadResult:
    """Load ``<task_id>.json`` from a hint directory."""
    return load_hint_file(Path(hints_directory) / f"{task_id}.json")


def load_hints_jsonl(path: PathLike) -> Dict[str, Hint]:
    """Load a versioned JSONL collection of complete hints by task ID."""
    hints: Dict[str, Hint] = {}
    for record in read_jsonl(path):
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("every hint JSONL record requires a non-empty task_id.")
        if task_id in hints:
            raise ValueError(f"duplicate hint task_id: {task_id!r}.")
        schema_version = record.get("schema_version")
        if schema_version != 1:
            raise ValueError(
                f"unsupported hint schema_version for {task_id!r}: {schema_version!r}."
            )
        hints[task_id] = Hint.from_mapping(record)
    return hints
