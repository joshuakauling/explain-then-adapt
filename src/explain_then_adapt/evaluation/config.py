"""Typed settings for ARC scoring and thesis compute accounting."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml  # type: ignore[import-untyped]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


@dataclass(frozen=True)
class GridParserSettings:
    max_height: int
    max_width: int

    def __post_init__(self) -> None:
        for value, name in (
            (self.max_height, "grid_parser.max_height"),
            (self.max_width, "grid_parser.max_width"),
        ):
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True)
class ComputeSettings:
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    training_token_multiplier: float

    def __post_init__(self) -> None:
        for value, name in (
            (self.prefill_tokens_per_second, "compute.prefill_tokens_per_second"),
            (self.decode_tokens_per_second, "compute.decode_tokens_per_second"),
            (self.training_token_multiplier, "compute.training_token_multiplier"),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number.")


@dataclass(frozen=True)
class EvaluationConfig:
    schema_version: int
    grid_parser: GridParserSettings
    compute: ComputeSettings

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported evaluation config schema version.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Load the versioned parser limits and thesis compute constants."""
    with path.open("r", encoding="utf-8") as file:
        root = _mapping(yaml.safe_load(file), "configuration")
    grid_parser = _mapping(root.get("grid_parser"), "grid_parser")
    compute = _mapping(root.get("compute"), "compute")
    return EvaluationConfig(
        schema_version=int(root["schema_version"]),
        grid_parser=GridParserSettings(
            max_height=int(grid_parser["max_height"]),
            max_width=int(grid_parser["max_width"]),
        ),
        compute=ComputeSettings(
            prefill_tokens_per_second=float(
                compute["prefill_tokens_per_second"]
            ),
            decode_tokens_per_second=float(compute["decode_tokens_per_second"]),
            training_token_multiplier=float(compute["training_token_multiplier"]),
        ),
    )
