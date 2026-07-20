"""Exact RM and PM prompt construction plus guidance artifact loading."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from explain_then_adapt.arc.formatting import (
    format_grid_to_string,
    format_puzzle_to_string,
)
from explain_then_adapt.arc.io import load_task
from explain_then_adapt.arc.transforms import (
    transform_individual_grid,
    transform_pairs,
)
from explain_then_adapt.arc.types import Task

from .config import InferenceConfig
from .planning import (
    InferenceVariant,
    PredictionRequest,
    ReasoningRequest,
)

GUIDANCE_HEADINGS = (
    "General natural language description:",
    "General steps:",
)


def prompt_sha256(prompt: str) -> str:
    """Hash a rendered prompt without storing it in large run artifacts."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def extract_guidance(trace: str) -> str:
    """Extract and validate the compact description after the final think block."""
    if not isinstance(trace, str):
        raise TypeError("reasoning trace must be a string.")
    _, marker, guidance = trace.rpartition("</think>")
    guidance = guidance.strip()
    if not marker or not guidance:
        raise ValueError("reasoning trace has no guidance after </think>.")
    return validate_guidance(guidance)


def validate_guidance(guidance: str) -> str:
    """Validate the two compact sections consumed by the Prediction Model."""
    if not isinstance(guidance, str):
        raise TypeError("guidance must be a string.")
    normalized = guidance.strip()
    if not normalized:
        raise ValueError("guidance must not be empty.")
    for heading in GUIDANCE_HEADINGS:
        if heading not in normalized:
            raise ValueError(f"guidance is missing {heading!r}.")
    return normalized


def _guidance_from_value(value: Any, key: str) -> str:
    if isinstance(value, Mapping):
        for field in ("guidance", "trace", "normalized_output", "raw_output"):
            candidate = value.get(field)
            if isinstance(candidate, str):
                return _guidance_from_value(candidate, key)
        error = value.get("validation_error")
        if isinstance(error, str):
            raise ValueError(f"guidance entry {key!r} is invalid: {error}")
        raise ValueError(f"guidance entry {key!r} contains no usable text.")
    if isinstance(value, list):
        if not value:
            raise ValueError(f"guidance entry {key!r} is an empty list.")
        # Historical RM artifacts contain n samples, while downstream code
        # always selected rm_index=0. New runs never write lists.
        return _guidance_from_value(value[0], key)
    if not isinstance(value, str):
        raise ValueError(f"guidance entry {key!r} must resolve to a string.")
    if "</think>" in value:
        return extract_guidance(value)
    return validate_guidance(value)


def _load_jsonl_guidance(path: Path) -> Dict[str, str]:
    guidance: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} must contain an object.")
            raw_key = value.get("request_id", value.get("key"))
            if not isinstance(raw_key, str) or not raw_key:
                raise ValueError(f"{path}:{line_number} has no non-empty request_id.")
            if raw_key in guidance:
                raise ValueError(f"duplicate guidance key {raw_key!r} in {path}.")
            guidance[raw_key] = _guidance_from_value(value, raw_key)
    if not guidance:
        raise ValueError(f"guidance artifact contains no records: {path}.")
    return guidance


def load_guidance(path: Path) -> Dict[str, str]:
    """Load new JSONL guidance or historical JSON RM output mappings."""
    if not path.is_file():
        raise FileNotFoundError(f"guidance artifact does not exist: {path}.")
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl_guidance(path)

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, Mapping):
        raise ValueError("guidance JSON must contain a mapping.")
    for wrapper in ("guidance", "guided", "predictions"):
        candidate = payload.get(wrapper)
        if isinstance(candidate, Mapping):
            payload = candidate
            break

    guidance: Dict[str, str] = {}
    for key, value in payload.items():
        if key == "meta":
            continue
        key_string = str(key)
        if key_string in guidance:
            raise ValueError(f"duplicate guidance key {key_string!r}.")
        guidance[key_string] = _guidance_from_value(value, key_string)
    if not guidance:
        raise ValueError(f"guidance artifact contains no records: {path}.")
    return guidance


def require_guidance_for_variants(
    variants: Sequence[InferenceVariant],
    guidance_by_key: Mapping[str, str],
) -> None:
    """Fail before loading a model when any required exact-key guidance is absent."""
    required = {variant.key for variant in variants}
    missing = sorted(key for key in required if key not in guidance_by_key)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"guidance is missing {len(missing)} required task views; "
            f"first keys: {preview}."
        )


def _transformed_train_pairs(
    task: Task,
    variant: InferenceVariant,
) -> Any:
    return transform_pairs(
        task["train"],
        variant.transformation_code,
        variant.value_mapping,
        variant.order_mapping,
    )


def build_reasoning_prompt_from_task(
    request: ReasoningRequest,
    task: Task,
    config: InferenceConfig,
) -> str:
    """Render demonstrations as the sole user message seen by the RM."""
    pairs = _transformed_train_pairs(task, request.variant)
    return format_puzzle_to_string(
        pairs,
        delimiter=config.serialization.grid_delimiter,
    )


def build_reasoning_prompt(
    request: ReasoningRequest,
    *,
    tasks_directory: Path,
    config: InferenceConfig,
) -> str:
    return build_reasoning_prompt_from_task(
        request,
        load_task(request.variant.task_id, tasks_directory),
        config,
    )


def _chat_block(role_header: str, content: str, message_end: str) -> str:
    return f"{role_header}{content}{message_end}"


def build_prediction_prompt_from_task(
    request: PredictionRequest,
    task: Task,
    config: InferenceConfig,
    *,
    guidance: Optional[str],
) -> str:
    """Render the exact raw Qwen conversation used by PM training."""
    if request.test_index >= len(task["test"]):
        raise IndexError(
            f"test index {request.test_index} is out of range for "
            f"{request.variant.task_id!r}."
        )
    serialization = config.serialization
    chunks = []
    if guidance is not None:
        chunks.append(
            _chat_block(
                serialization.system_header,
                validate_guidance(guidance),
                serialization.message_end,
            )
        )

    for pair in _transformed_train_pairs(task, request.variant):
        input_text = format_grid_to_string(
            pair["input"],
            delimiter=serialization.grid_delimiter,
        )
        output_text = format_grid_to_string(
            pair["output"],
            delimiter=serialization.grid_delimiter,
        )
        chunks.append(
            _chat_block(
                serialization.user_header,
                input_text,
                serialization.message_end,
            )
        )
        chunks.append(
            _chat_block(
                serialization.assistant_header,
                output_text,
                serialization.message_end,
            )
        )

    test_input = transform_individual_grid(
        task["test"][request.test_index]["input"],
        request.variant.transformation_code,
        request.variant.value_mapping,
    )
    chunks.append(
        _chat_block(
            serialization.user_header,
            format_grid_to_string(
                test_input,
                delimiter=serialization.grid_delimiter,
            ),
            serialization.message_end,
        )
    )
    chunks.append(serialization.assistant_header)
    return "".join(chunks)


def build_prediction_prompt(
    request: PredictionRequest,
    *,
    tasks_directory: Path,
    config: InferenceConfig,
    guidance: Optional[str],
) -> str:
    return build_prediction_prompt_from_task(
        request,
        load_task(request.variant.task_id, tasks_directory),
        config,
        guidance=guidance,
    )
