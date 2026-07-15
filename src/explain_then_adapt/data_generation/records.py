"""Provider-independent records for reasoning-data generation runs."""

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from explain_then_adapt.arc.augmented_keys import parse_order_mapping
from explain_then_adapt.arc.transforms import parse_value_mapping
from explain_then_adapt.arc.types import TRANSFORM_CODES


PathLike = Union[str, Path]


class GenerationStage(str, Enum):
    """Supported stages of the reasoning-data pipeline."""

    INITIAL = "initial"
    JUDGE = "judge"
    REWRITE = "rewrite"


class HintMode(str, Enum):
    """Whether a generation request contains a complete manual hint."""

    PROVIDED = "provided"
    NONE = "none"


@dataclass(frozen=True)
class ChatMessage:
    """One provider-independent chat message."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError("role must be 'system', 'user', or 'assistant'.")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("message content must be a non-empty string.")

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChatMessage":
        return cls(role=str(value["role"]), content=str(value["content"]))


@dataclass(frozen=True)
class SamplingParameters:
    """Sampling settings shared by remote and local execution backends."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 8192
    seed: Optional[int] = None
    use_provider_defaults: bool = False

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative.")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1].")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")

    def to_dict(self) -> Dict[str, Any]:
        if self.use_provider_defaults:
            return {"use_provider_defaults": True}
        result: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            result["seed"] = self.seed
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SamplingParameters":
        return cls(
            temperature=float(value.get("temperature", 0.0)),
            top_p=float(value.get("top_p", 1.0)),
            max_tokens=int(value.get("max_tokens", 8192)),
            seed=int(value["seed"]) if value.get("seed") is not None else None,
            use_provider_defaults=bool(value.get("use_provider_defaults", False)),
        )


@dataclass(frozen=True)
class TokenUsage:
    """Normalized token counts reported by an execution backend."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TokenUsage":
        return cls(
            prompt_tokens=int(value.get("prompt_tokens", 0)),
            completion_tokens=int(value.get("completion_tokens", 0)),
        )


@dataclass(frozen=True)
class AugmentationSpec:
    """Structured identity of one trace-aware augmentation."""

    source_trace_id: str
    transformation_code: str
    value_mapping: str
    order_mapping: str
    style: str = "neutral"
    variant_index: int = 0

    def __post_init__(self) -> None:
        if not self.source_trace_id.strip():
            raise ValueError("source_trace_id must not be empty.")
        if self.transformation_code not in TRANSFORM_CODES:
            raise ValueError(
                f"transformation_code must be one of {TRANSFORM_CODES}."
            )
        parse_value_mapping(self.value_mapping)
        parse_order_mapping(self.order_mapping)
        if not self.style.strip():
            raise ValueError("style must not be empty.")
        if self.variant_index < 0:
            raise ValueError("variant_index must be non-negative.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_trace_id": self.source_trace_id,
            "transformation_code": self.transformation_code,
            "value_mapping": self.value_mapping,
            "order_mapping": self.order_mapping,
            "style": self.style,
            "variant_index": self.variant_index,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AugmentationSpec":
        return cls(
            source_trace_id=str(value["source_trace_id"]),
            transformation_code=str(value["transformation_code"]),
            value_mapping=str(value["value_mapping"]),
            order_mapping=str(value["order_mapping"]),
            style=str(value.get("style", "neutral")),
            variant_index=int(value.get("variant_index", 0)),
        )


def make_request_id(
    stage: GenerationStage,
    task_id: str,
    identity: Mapping[str, Any],
) -> str:
    """Build a stable request identifier from the semantic request identity."""
    payload = json.dumps(
        {"stage": stage.value, "task_id": task_id, "identity": identity},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{stage.value}-{task_id}-{digest}"


@dataclass(frozen=True)
class GenerationRequest:
    """A complete model request independent of any provider SDK."""

    request_id: str
    task_id: str
    stage: GenerationStage
    messages: Tuple[ChatMessage, ...]
    prompt_version: str
    hint_mode: HintMode = HintMode.NONE
    few_shot_task_ids: Tuple[str, ...] = ()
    sampling: SamplingParameters = field(default_factory=SamplingParameters)
    augmentation: Optional[AugmentationSpec] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.task_id.strip():
            raise ValueError("request_id and task_id must not be empty.")
        if not self.messages:
            raise ValueError("a generation request requires at least one message.")
        if not self.prompt_version.strip():
            raise ValueError("prompt_version must not be empty.")
        if self.stage is GenerationStage.REWRITE and self.augmentation is None:
            raise ValueError("rewrite requests require an augmentation specification.")

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "stage": self.stage.value,
            "messages": [message.to_dict() for message in self.messages],
            "prompt_version": self.prompt_version,
            "hint_mode": self.hint_mode.value,
            "few_shot_task_ids": list(self.few_shot_task_ids),
            "sampling": self.sampling.to_dict(),
            "metadata": dict(self.metadata),
        }
        if self.augmentation is not None:
            result["augmentation"] = self.augmentation.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenerationRequest":
        messages = value.get("messages")
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise ValueError("request messages must be a list.")
        augmentation = value.get("augmentation")
        return cls(
            request_id=str(value["request_id"]),
            task_id=str(value["task_id"]),
            stage=GenerationStage(str(value["stage"])),
            messages=tuple(ChatMessage.from_dict(message) for message in messages),
            prompt_version=str(value["prompt_version"]),
            hint_mode=HintMode(str(value.get("hint_mode", HintMode.NONE.value))),
            few_shot_task_ids=tuple(
                str(task_id) for task_id in value.get("few_shot_task_ids", [])
            ),
            sampling=SamplingParameters.from_dict(value.get("sampling", {})),
            augmentation=(
                AugmentationSpec.from_dict(augmentation)
                if isinstance(augmentation, Mapping)
                else None
            ),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class GenerationResult:
    """Normalized output produced by a generation backend."""

    request_id: str
    task_id: str
    stage: GenerationStage
    backend: str
    model: str
    prompt_version: Optional[str] = None
    hint_mode: Optional[HintMode] = None
    few_shot_task_ids: Tuple[str, ...] = ()
    sampling: Optional[SamplingParameters] = None
    augmentation: Optional[AugmentationSpec] = None
    raw_output: str = ""
    normalized_output: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    error: Optional[str] = None
    validation: Optional[Mapping[str, Any]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "stage": self.stage.value,
            "backend": self.backend,
            "model": self.model,
            "raw_output": self.raw_output,
            "usage": self.usage.to_dict(),
            "metadata": dict(self.metadata),
        }
        if self.prompt_version is not None:
            result["prompt_version"] = self.prompt_version
        if self.hint_mode is not None:
            result["hint_mode"] = self.hint_mode.value
        if self.few_shot_task_ids:
            result["few_shot_task_ids"] = list(self.few_shot_task_ids)
        if self.sampling is not None:
            result["sampling"] = self.sampling.to_dict()
        if self.augmentation is not None:
            result["augmentation"] = self.augmentation.to_dict()
        for key, value in (
            ("normalized_output", self.normalized_output),
            ("finish_reason", self.finish_reason),
            ("error", self.error),
        ):
            if value is not None:
                result[key] = value
        if self.validation is not None:
            result["validation"] = dict(self.validation)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenerationResult":
        return cls(
            request_id=str(value["request_id"]),
            task_id=str(value["task_id"]),
            stage=GenerationStage(str(value["stage"])),
            backend=str(value["backend"]),
            model=str(value["model"]),
            prompt_version=(
                str(value["prompt_version"])
                if value.get("prompt_version") is not None
                else None
            ),
            hint_mode=(
                HintMode(str(value["hint_mode"]))
                if value.get("hint_mode") is not None
                else None
            ),
            few_shot_task_ids=tuple(
                str(task_id) for task_id in value.get("few_shot_task_ids", [])
            ),
            sampling=(
                SamplingParameters.from_dict(value["sampling"])
                if isinstance(value.get("sampling"), Mapping)
                else None
            ),
            augmentation=(
                AugmentationSpec.from_dict(value["augmentation"])
                if isinstance(value.get("augmentation"), Mapping)
                else None
            ),
            raw_output=str(value.get("raw_output", "")),
            normalized_output=(
                str(value["normalized_output"])
                if value.get("normalized_output") is not None
                else None
            ),
            finish_reason=(
                str(value["finish_reason"])
                if value.get("finish_reason") is not None
                else None
            ),
            usage=TokenUsage.from_dict(value.get("usage", {})),
            error=str(value["error"]) if value.get("error") is not None else None,
            validation=(
                dict(value["validation"])
                if isinstance(value.get("validation"), Mapping)
                else None
            ),
            metadata=dict(value.get("metadata", {})),
        )


def write_jsonl(records: Iterable[Mapping[str, Any]], path: PathLike) -> int:
    """Write mappings as UTF-8 JSONL and return the number of records."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    """Read non-empty UTF-8 JSONL lines into dictionaries."""
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number} of {path}.") from error
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} of {path} must contain an object.")
            records.append(value)
    return records


def write_requests(requests: Iterable[GenerationRequest], path: PathLike) -> int:
    return write_jsonl((request.to_dict() for request in requests), path)


def read_requests(path: PathLike) -> List[GenerationRequest]:
    return [GenerationRequest.from_dict(value) for value in read_jsonl(path)]


def write_results(results: Iterable[GenerationResult], path: PathLike) -> int:
    return write_jsonl((result.to_dict() for result in results), path)


def read_results(path: PathLike) -> List[GenerationResult]:
    return [GenerationResult.from_dict(value) for value in read_jsonl(path)]
