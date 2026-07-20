"""Typed configuration for the final thesis inference protocols."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import yaml  # type: ignore[import-untyped]

INFERENCE_PROTOCOLS: Tuple[str, ...] = (
    "standard32",
    "augmented64",
    "budgeted64",
)
GUIDANCE_MODES: Tuple[str, ...] = ("guided", "unguided")
GUIDANCE_BUDGETS: Tuple[int, ...] = (0, 8, 16, 32, 64)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True)
class SamplingSettings:
    temperature: float
    top_p: float
    max_tokens: int

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("sampling temperature must be non-negative.")
        if not 0 < self.top_p <= 1:
            raise ValueError("sampling top_p must be in (0, 1].")
        _positive_integer(self.max_tokens, "sampling max_tokens")


@dataclass(frozen=True)
class EngineSettings:
    dtype: str
    gpu_memory_utilization: float
    max_model_len: int
    tensor_parallel_size: int
    request_batch_size: int

    def __post_init__(self) -> None:
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("engine dtype must be bfloat16, float16, or float32.")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("engine gpu_memory_utilization must be in (0, 1].")
        _positive_integer(self.max_model_len, "engine max_model_len")
        _positive_integer(
            self.tensor_parallel_size,
            "engine tensor_parallel_size",
        )
        _positive_integer(self.request_batch_size, "engine request_batch_size")


@dataclass(frozen=True)
class PredictionEngineSettings(EngineSettings):
    ttt_max_model_len: int
    max_lora_rank: int

    def __post_init__(self) -> None:
        super().__post_init__()
        _positive_integer(self.ttt_max_model_len, "engine ttt_max_model_len")
        if self.ttt_max_model_len < self.max_model_len:
            raise ValueError(
                "engine ttt_max_model_len cannot be shorter than max_model_len."
            )
        if self.max_lora_rank not in {8, 16, 32, 64, 128, 256}:
            raise ValueError("engine max_lora_rank is unsupported by this pipeline.")


@dataclass(frozen=True)
class ReasoningSettings:
    sampling: SamplingSettings
    engine: EngineSettings


@dataclass(frozen=True)
class PredictionSettings:
    sampling: SamplingSettings
    engine: PredictionEngineSettings
    stop: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.stop or any(not value for value in self.stop):
            raise ValueError("prediction stop must contain non-empty strings.")


@dataclass(frozen=True)
class SerializationSettings:
    grid_delimiter: str
    system_header: str
    user_header: str
    assistant_header: str
    message_end: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.system_header, "serialization.system_header"),
            (self.user_header, "serialization.user_header"),
            (self.assistant_header, "serialization.assistant_header"),
            (self.message_end, "serialization.message_end"),
        ):
            if not value:
                raise ValueError(f"{name} must not be empty.")


@dataclass(frozen=True)
class ProtocolSettings:
    standard_samples: int
    augmented_variants: int
    augmented_samples_per_variant: int
    budgeted_total_samples: int
    guidance_budgets: Tuple[int, ...]

    def __post_init__(self) -> None:
        if self.standard_samples != 32:
            raise ValueError("the final standard protocol requires 32 samples.")
        if self.augmented_variants != 64:
            raise ValueError("the final augmented protocol requires 64 variants.")
        if self.augmented_samples_per_variant != 1:
            raise ValueError(
                "the final augmented protocol requires one sample per variant."
            )
        if self.budgeted_total_samples != 64:
            raise ValueError("the final budgeted protocol requires 64 samples.")
        if self.guidance_budgets != GUIDANCE_BUDGETS:
            raise ValueError(
                f"protocol guidance_budgets must be exactly {GUIDANCE_BUDGETS}."
            )


@dataclass(frozen=True)
class InferenceConfig:
    schema_version: int
    augmentation_seed: int
    sampling_seed: int
    variants_per_transform: int
    reasoning: ReasoningSettings
    prediction: PredictionSettings
    serialization: SerializationSettings
    protocols: ProtocolSettings

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported inference config schema version.")
        for value, name in (
            (self.augmentation_seed, "augmentation_seed"),
            (self.sampling_seed, "sampling_seed"),
        ):
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.variants_per_transform != 8:
            raise ValueError(
                "the final inference protocol requires eight variants per transform."
            )
        if self.protocols.augmented_variants != 8 * self.variants_per_transform:
            raise ValueError("augmentation and protocol variant counts disagree.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sampling(value: Mapping[str, Any]) -> SamplingSettings:
    return SamplingSettings(
        temperature=float(value["temperature"]),
        top_p=float(value["top_p"]),
        max_tokens=int(value["max_tokens"]),
    )


def _engine(value: Mapping[str, Any]) -> EngineSettings:
    return EngineSettings(
        dtype=str(value["dtype"]),
        gpu_memory_utilization=float(value["gpu_memory_utilization"]),
        max_model_len=int(value["max_model_len"]),
        tensor_parallel_size=int(value["tensor_parallel_size"]),
        request_batch_size=int(value["request_batch_size"]),
    )


def load_inference_config(path: Path) -> InferenceConfig:
    """Load and validate the versioned inference configuration."""
    with path.open("r", encoding="utf-8") as file:
        root = _mapping(yaml.safe_load(file), "configuration")

    augmentation = _mapping(root.get("augmentation"), "augmentation")
    reasoning = _mapping(root.get("reasoning"), "reasoning")
    prediction = _mapping(root.get("prediction"), "prediction")
    reasoning_sampling = _mapping(reasoning.get("sampling"), "reasoning.sampling")
    reasoning_engine = _mapping(reasoning.get("engine"), "reasoning.engine")
    prediction_sampling = _mapping(
        prediction.get("sampling"),
        "prediction.sampling",
    )
    prediction_engine = _mapping(prediction.get("engine"), "prediction.engine")
    serialization = _mapping(root.get("serialization"), "serialization")
    protocols = _mapping(root.get("protocols"), "protocols")

    base_prediction_engine = _engine(prediction_engine)
    return InferenceConfig(
        schema_version=int(root["schema_version"]),
        augmentation_seed=int(root["augmentation_seed"]),
        sampling_seed=int(root["sampling_seed"]),
        variants_per_transform=int(augmentation["variants_per_transform"]),
        reasoning=ReasoningSettings(
            sampling=_sampling(reasoning_sampling),
            engine=_engine(reasoning_engine),
        ),
        prediction=PredictionSettings(
            sampling=_sampling(prediction_sampling),
            engine=PredictionEngineSettings(
                dtype=base_prediction_engine.dtype,
                gpu_memory_utilization=(base_prediction_engine.gpu_memory_utilization),
                max_model_len=base_prediction_engine.max_model_len,
                tensor_parallel_size=base_prediction_engine.tensor_parallel_size,
                request_batch_size=base_prediction_engine.request_batch_size,
                ttt_max_model_len=int(prediction_engine["ttt_max_model_len"]),
                max_lora_rank=int(prediction_engine["max_lora_rank"]),
            ),
            stop=tuple(str(value) for value in prediction["stop"]),
        ),
        serialization=SerializationSettings(
            grid_delimiter=str(serialization["grid_delimiter"]),
            system_header=str(serialization["system_header"]),
            user_header=str(serialization["user_header"]),
            assistant_header=str(serialization["assistant_header"]),
            message_end=str(serialization["message_end"]),
        ),
        protocols=ProtocolSettings(
            standard_samples=int(protocols["standard_samples"]),
            augmented_variants=int(protocols["augmented_variants"]),
            augmented_samples_per_variant=int(
                protocols["augmented_samples_per_variant"]
            ),
            budgeted_total_samples=int(protocols["budgeted_total_samples"]),
            guidance_budgets=tuple(
                int(value) for value in protocols["guidance_budgets"]
            ),
        ),
    )
