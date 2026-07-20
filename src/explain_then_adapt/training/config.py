"""Typed configuration for offline model training and online adaptation."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml  # type: ignore[import-untyped]

TTT_GUIDANCE_BUDGETS: Tuple[int, ...] = (0, 8, 16, 32, 64)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _positive(value: int, name: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True)
class ModelSettings:
    name: str
    revision: Optional[str]
    quantization_bits: int
    quantization_type: str
    double_quantization: bool
    dtype: str
    attention_implementation: str
    gradient_checkpointing: bool

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("model.name must not be empty.")
        if self.quantization_bits not in {4, 8}:
            raise ValueError("model.quantization_bits must be 4 or 8.")
        if self.quantization_type not in {"nf4", "fp4"}:
            raise ValueError("model.quantization_type must be nf4 or fp4.")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("model.dtype must be bfloat16, float16, or float32.")
        if not self.attention_implementation.strip():
            raise ValueError("model.attention_implementation must not be empty.")


@dataclass(frozen=True)
class LoraSettings:
    target_modules: Tuple[str, ...]
    rank: int
    alpha: int
    dropout: float
    bias: str
    use_rslora: bool
    use_dora: bool

    def __post_init__(self) -> None:
        if not self.target_modules or any(
            not item.strip() for item in self.target_modules
        ):
            raise ValueError("lora.target_modules must contain non-empty names.")
        _positive(self.rank, "lora.rank")
        _positive(self.alpha, "lora.alpha")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("lora.dropout must be in [0, 1).")
        if self.bias not in {"none", "all", "lora_only"}:
            raise ValueError("lora.bias must be none, all, or lora_only.")


@dataclass(frozen=True)
class DataSettings:
    variants_per_task: int
    max_sequence_length: int
    grid_delimiter: str
    assistant_header: str
    pad_to_multiple_of: int

    def __post_init__(self) -> None:
        _positive(self.variants_per_task, "data.variants_per_task")
        _positive(self.max_sequence_length, "data.max_sequence_length")
        if not self.assistant_header:
            raise ValueError("data.assistant_header must not be empty.")
        _positive(self.pad_to_multiple_of, "data.pad_to_multiple_of")


@dataclass(frozen=True)
class OptimizationSettings:
    optimizer: str
    scheduler: str
    epochs: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    validation_batch_size: int
    peak_learning_rate: float
    end_learning_rate: float
    warmup_ratio: float
    warmup_start_factor: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    weight_decay: float
    max_grad_norm: float

    def __post_init__(self) -> None:
        if self.optimizer != "adamw_8bit":
            raise ValueError("optimization.optimizer must be adamw_8bit.")
        if self.scheduler != "cosine":
            raise ValueError("optimization.scheduler must be cosine.")
        _positive(self.epochs, "optimization.epochs")
        _positive(self.micro_batch_size, "optimization.micro_batch_size")
        _positive(
            self.gradient_accumulation_steps,
            "optimization.gradient_accumulation_steps",
        )
        _positive(self.validation_batch_size, "optimization.validation_batch_size")
        if self.peak_learning_rate <= 0:
            raise ValueError("optimization.peak_learning_rate must be positive.")
        if not 0 < self.end_learning_rate <= self.peak_learning_rate:
            raise ValueError(
                "optimization.end_learning_rate must be positive and no greater "
                "than peak_learning_rate."
            )
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("optimization.warmup_ratio must be in [0, 1).")
        if not 0 < self.warmup_start_factor <= 1:
            raise ValueError("optimization.warmup_start_factor must be in (0, 1].")
        if not 0 <= self.adam_beta1 < 1 or not 0 <= self.adam_beta2 < 1:
            raise ValueError("optimization Adam betas must be in [0, 1).")
        if self.adam_epsilon <= 0:
            raise ValueError("optimization.adam_epsilon must be positive.")
        if self.weight_decay < 0:
            raise ValueError("optimization.weight_decay must be non-negative.")
        if self.max_grad_norm <= 0:
            raise ValueError("optimization.max_grad_norm must be positive.")


@dataclass(frozen=True)
class ControlSettings:
    log_every_steps: int
    validate_every_steps: int
    checkpoint_every_steps: int
    checkpoint_epochs: Tuple[int, ...]
    checkpoint_steps: Tuple[int, ...]

    def __post_init__(self) -> None:
        _positive(self.log_every_steps, "control.log_every_steps")
        _positive(self.validate_every_steps, "control.validate_every_steps")
        _positive(self.checkpoint_every_steps, "control.checkpoint_every_steps")
        if any(
            isinstance(epoch, bool) or epoch <= 0 for epoch in self.checkpoint_epochs
        ):
            raise ValueError(
                "control.checkpoint_epochs must contain positive integers."
            )
        if len(set(self.checkpoint_epochs)) != len(self.checkpoint_epochs):
            raise ValueError("control.checkpoint_epochs must not contain duplicates.")
        if any(isinstance(step, bool) or step <= 0 for step in self.checkpoint_steps):
            raise ValueError("control.checkpoint_steps must contain positive integers.")
        if len(set(self.checkpoint_steps)) != len(self.checkpoint_steps):
            raise ValueError("control.checkpoint_steps must not contain duplicates.")


@dataclass(frozen=True)
class LoaderSettings:
    num_workers: int
    pin_memory: bool
    persistent_workers: bool

    def __post_init__(self) -> None:
        if isinstance(self.num_workers, bool) or self.num_workers < 0:
            raise ValueError("loader.num_workers must be a non-negative integer.")
        if self.persistent_workers:
            raise ValueError(
                "loader.persistent_workers must be false because the selected "
                "training variant changes between epochs."
            )


@dataclass(frozen=True)
class TrackingSettings:
    wandb_project: str

    def __post_init__(self) -> None:
        if not self.wandb_project.strip():
            raise ValueError("tracking.wandb_project must not be empty.")


@dataclass(frozen=True)
class ReasoningTrainingConfig:
    schema_version: int
    seed: int
    model: ModelSettings
    lora: LoraSettings
    data: DataSettings
    optimization: OptimizationSettings
    control: ControlSettings
    loader: LoaderSettings
    tracking: TrackingSettings

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported Reasoning Model config schema version.")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if self.optimization.epochs > self.data.variants_per_task:
            raise ValueError(
                "optimization.epochs cannot exceed data.variants_per_task when "
                "variants are consumed without repetition."
            )
        if any(
            epoch > self.optimization.epochs for epoch in self.control.checkpoint_epochs
        ):
            raise ValueError(
                "control.checkpoint_epochs cannot exceed optimization.epochs."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionDataSettings:
    variants_per_task: int
    max_sequence_length: int
    grid_delimiter: str
    system_header: str
    user_header: str
    assistant_header: str
    message_end: str
    pad_to_multiple_of: int
    synthetic_task_count: int
    rearc_task_count: int
    rearc_examples_per_task: int
    rearc_pairs_per_variant: int

    def __post_init__(self) -> None:
        _positive(self.variants_per_task, "data.variants_per_task")
        _positive(self.max_sequence_length, "data.max_sequence_length")
        for value, name in (
            (self.system_header, "data.system_header"),
            (self.user_header, "data.user_header"),
            (self.assistant_header, "data.assistant_header"),
            (self.message_end, "data.message_end"),
        ):
            if not value:
                raise ValueError(f"{name} must not be empty.")
        _positive(self.pad_to_multiple_of, "data.pad_to_multiple_of")
        _positive(self.synthetic_task_count, "data.synthetic_task_count")
        _positive(self.rearc_task_count, "data.rearc_task_count")
        _positive(self.rearc_examples_per_task, "data.rearc_examples_per_task")
        _positive(self.rearc_pairs_per_variant, "data.rearc_pairs_per_variant")


@dataclass(frozen=True)
class PredictionProfileSettings:
    name: str
    data_source: str
    guided: bool
    ignore_first_response: bool
    initialization: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("prediction profile name must not be empty.")
        if self.data_source not in {"synthetic", "rearc"}:
            raise ValueError(
                "prediction profile data_source must be synthetic or rearc."
            )
        if self.initialization not in {"pretrained", "merged_model"}:
            raise ValueError(
                "prediction profile initialization must be pretrained or merged_model."
            )
        if self.data_source == "rearc" and self.guided:
            raise ValueError(
                "the ReARC source does not provide task-specific guidance."
            )
        if self.initialization == "merged_model" and self.data_source != "synthetic":
            raise ValueError(
                "merged-model initialization is only defined for synthetic data."
            )


@dataclass(frozen=True)
class PredictionTrainingConfig:
    schema_version: int
    seed: int
    model: ModelSettings
    lora: LoraSettings
    data: PredictionDataSettings
    optimization: OptimizationSettings
    control: ControlSettings
    loader: LoaderSettings
    tracking: TrackingSettings
    profile: PredictionProfileSettings

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported Prediction Model config schema version.")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if self.optimization.epochs > self.data.variants_per_task:
            raise ValueError(
                "optimization.epochs cannot exceed data.variants_per_task when "
                "variants are consumed without repetition."
            )
        if any(
            epoch > self.optimization.epochs for epoch in self.control.checkpoint_epochs
        ):
            raise ValueError(
                "control.checkpoint_epochs cannot exceed optimization.epochs."
            )

        micro_batches = (
            self.expected_task_count + self.optimization.micro_batch_size - 1
        ) // self.optimization.micro_batch_size
        steps_per_epoch = (
            micro_batches + self.optimization.gradient_accumulation_steps - 1
        ) // self.optimization.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.optimization.epochs
        if any(step > total_steps for step in self.control.checkpoint_steps):
            raise ValueError(
                "control.checkpoint_steps cannot exceed the resolved run length."
            )

    @property
    def expected_task_count(self) -> int:
        if self.profile.data_source == "synthetic":
            return self.data.synthetic_task_count
        return self.data.rearc_task_count

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TTTDataSettings:
    variants_per_transform: int
    max_sequence_length: int
    grid_delimiter: str
    system_header: str
    user_header: str
    assistant_header: str
    message_end: str
    pad_to_multiple_of: int
    ignore_first_response: bool
    empty_guidance_content: str
    missing_guidance_policy: str

    def __post_init__(self) -> None:
        if self.variants_per_transform != 8:
            raise ValueError(
                "data.variants_per_transform must be 8 for the final 64-variant "
                "TTT protocol."
            )
        _positive(self.max_sequence_length, "data.max_sequence_length")
        for value, name in (
            (self.system_header, "data.system_header"),
            (self.user_header, "data.user_header"),
            (self.assistant_header, "data.assistant_header"),
            (self.message_end, "data.message_end"),
        ):
            if not value:
                raise ValueError(f"{name} must not be empty.")
        _positive(self.pad_to_multiple_of, "data.pad_to_multiple_of")
        if not self.ignore_first_response:
            raise ValueError("data.ignore_first_response must be true for final TTT.")
        if self.empty_guidance_content != " ":
            raise ValueError(
                "data.empty_guidance_content must be exactly one ASCII space."
            )
        if self.missing_guidance_policy not in {"error", "omit_system"}:
            raise ValueError(
                "data.missing_guidance_policy must be error or omit_system."
            )

    @property
    def variants_per_task(self) -> int:
        return 8 * self.variants_per_transform


@dataclass(frozen=True)
class TTTOptimizationSettings:
    optimizer: str
    scheduler: str
    epochs: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    warmup_start_learning_rate: float
    min_learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    weight_decay: float
    max_grad_norm: float

    def __post_init__(self) -> None:
        if self.optimizer != "adamw_8bit":
            raise ValueError("optimization.optimizer must be adamw_8bit.")
        if self.scheduler != "cosine":
            raise ValueError("optimization.scheduler must be cosine.")
        if self.epochs != 1:
            raise ValueError("optimization.epochs must be 1 for online TTT.")
        if self.micro_batch_size != 1:
            raise ValueError(
                "optimization.micro_batch_size must be 1 for the final TTT protocol."
            )
        if self.gradient_accumulation_steps != 1:
            raise ValueError(
                "optimization.gradient_accumulation_steps must be 1 for final TTT."
            )
        if self.warmup_ratio != 0.5:
            raise ValueError("optimization.warmup_ratio must be 0.5 for final TTT.")
        if self.warmup_start_learning_rate != 0.0:
            raise ValueError("optimization.warmup_start_learning_rate must be 0.0.")
        if self.min_learning_rate <= 0:
            raise ValueError("optimization.min_learning_rate must be positive.")
        if not 0 <= self.adam_beta1 < 1 or not 0 <= self.adam_beta2 < 1:
            raise ValueError("optimization Adam betas must be in [0, 1).")
        if self.adam_epsilon <= 0:
            raise ValueError("optimization.adam_epsilon must be positive.")
        if self.weight_decay < 0:
            raise ValueError("optimization.weight_decay must be non-negative.")
        if self.max_grad_norm <= 0:
            raise ValueError("optimization.max_grad_norm must be positive.")


@dataclass(frozen=True)
class TTTProfileSettings:
    name: str
    guidance_mode: str
    guidance_budget: int
    peak_learning_rate: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("TTT profile name must not be empty.")
        if self.guidance_mode not in {"guided", "unguided"}:
            raise ValueError("TTT profile guidance_mode must be guided or unguided.")
        if (
            isinstance(self.guidance_budget, bool)
            or self.guidance_budget not in TTT_GUIDANCE_BUDGETS
        ):
            raise ValueError(
                "TTT guidance_budget must be one of " f"{TTT_GUIDANCE_BUDGETS}."
            )
        if self.guidance_mode == "unguided" and self.guidance_budget != 0:
            raise ValueError("unguided TTT must use guidance budget 0.")
        if self.peak_learning_rate <= 0:
            raise ValueError("TTT peak_learning_rate must be positive.")


@dataclass(frozen=True)
class TTTTrainingConfig:
    schema_version: int
    seed: int
    model: ModelSettings
    lora: LoraSettings
    data: TTTDataSettings
    optimization: TTTOptimizationSettings
    loader: LoaderSettings
    profile: TTTProfileSettings

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported TTT config schema version.")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if self.optimization.min_learning_rate > self.profile.peak_learning_rate:
            raise ValueError(
                "optimization.min_learning_rate cannot exceed the profile peak."
            )
        if self.data.variants_per_task != 64:
            raise ValueError("the final TTT protocol must contain 64 updates per task.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_reasoning_training_config(path: Path) -> ReasoningTrainingConfig:
    """Load and validate one Reasoning Model YAML configuration."""
    with path.open("r", encoding="utf-8") as file:
        root = _mapping(yaml.safe_load(file), "configuration")

    model = _mapping(root.get("model"), "model")
    lora = _mapping(root.get("lora"), "lora")
    data = _mapping(root.get("data"), "data")
    optimization = _mapping(root.get("optimization"), "optimization")
    control = _mapping(root.get("control"), "control")
    loader = _mapping(root.get("loader"), "loader")
    tracking = _mapping(root.get("tracking"), "tracking")

    revision_value = model.get("revision")
    revision = str(revision_value) if revision_value is not None else None
    return ReasoningTrainingConfig(
        schema_version=int(root["schema_version"]),
        seed=int(root["seed"]),
        model=ModelSettings(
            name=str(model["name"]),
            revision=revision,
            quantization_bits=int(model["quantization_bits"]),
            quantization_type=str(model["quantization_type"]),
            double_quantization=bool(model["double_quantization"]),
            dtype=str(model["dtype"]),
            attention_implementation=str(model["attention_implementation"]),
            gradient_checkpointing=bool(model["gradient_checkpointing"]),
        ),
        lora=LoraSettings(
            target_modules=tuple(str(item) for item in lora["target_modules"]),
            rank=int(lora["rank"]),
            alpha=int(lora["alpha"]),
            dropout=float(lora["dropout"]),
            bias=str(lora["bias"]),
            use_rslora=bool(lora["use_rslora"]),
            use_dora=bool(lora["use_dora"]),
        ),
        data=DataSettings(
            variants_per_task=int(data["variants_per_task"]),
            max_sequence_length=int(data["max_sequence_length"]),
            grid_delimiter=str(data["grid_delimiter"]),
            assistant_header=str(data["assistant_header"]),
            pad_to_multiple_of=int(data["pad_to_multiple_of"]),
        ),
        optimization=OptimizationSettings(
            optimizer=str(optimization["optimizer"]),
            scheduler=str(optimization["scheduler"]),
            epochs=int(optimization["epochs"]),
            micro_batch_size=int(optimization["micro_batch_size"]),
            gradient_accumulation_steps=int(
                optimization["gradient_accumulation_steps"]
            ),
            validation_batch_size=int(optimization["validation_batch_size"]),
            peak_learning_rate=float(optimization["peak_learning_rate"]),
            end_learning_rate=float(optimization["end_learning_rate"]),
            warmup_ratio=float(optimization["warmup_ratio"]),
            warmup_start_factor=float(optimization["warmup_start_factor"]),
            adam_beta1=float(optimization["adam_beta1"]),
            adam_beta2=float(optimization["adam_beta2"]),
            adam_epsilon=float(optimization["adam_epsilon"]),
            weight_decay=float(optimization["weight_decay"]),
            max_grad_norm=float(optimization["max_grad_norm"]),
        ),
        control=ControlSettings(
            log_every_steps=int(control["log_every_steps"]),
            validate_every_steps=int(control["validate_every_steps"]),
            checkpoint_every_steps=int(control["checkpoint_every_steps"]),
            checkpoint_epochs=tuple(
                int(epoch) for epoch in control["checkpoint_epochs"]
            ),
            checkpoint_steps=tuple(
                int(step) for step in control.get("checkpoint_steps", ())
            ),
        ),
        loader=LoaderSettings(
            num_workers=int(loader["num_workers"]),
            pin_memory=bool(loader["pin_memory"]),
            persistent_workers=bool(loader["persistent_workers"]),
        ),
        tracking=TrackingSettings(
            wandb_project=str(tracking["wandb_project"]),
        ),
    )


def load_prediction_training_config(
    path: Path,
    profile_name: str,
) -> PredictionTrainingConfig:
    """Load and resolve one Prediction Model profile from the shared YAML file."""
    if not profile_name.strip():
        raise ValueError("profile_name must not be empty.")
    with path.open("r", encoding="utf-8") as file:
        root = _mapping(yaml.safe_load(file), "configuration")

    model = _mapping(root.get("model"), "model")
    lora = _mapping(root.get("lora"), "lora")
    data = _mapping(root.get("data"), "data")
    optimization = _mapping(root.get("optimization"), "optimization")
    control = _mapping(root.get("control"), "control")
    loader = _mapping(root.get("loader"), "loader")
    tracking = _mapping(root.get("tracking"), "tracking")
    profiles = _mapping(root.get("profiles"), "profiles")
    if profile_name not in profiles:
        available = ", ".join(sorted(str(name) for name in profiles))
        raise ValueError(
            f"unknown Prediction Model profile {profile_name!r}; "
            f"available profiles: {available}."
        )
    profile = _mapping(profiles[profile_name], f"profiles.{profile_name}")

    revision_value = model.get("revision")
    revision = str(revision_value) if revision_value is not None else None
    return PredictionTrainingConfig(
        schema_version=int(root["schema_version"]),
        seed=int(root["seed"]),
        model=ModelSettings(
            name=str(model["name"]),
            revision=revision,
            quantization_bits=int(model["quantization_bits"]),
            quantization_type=str(model["quantization_type"]),
            double_quantization=bool(model["double_quantization"]),
            dtype=str(model["dtype"]),
            attention_implementation=str(model["attention_implementation"]),
            gradient_checkpointing=bool(model["gradient_checkpointing"]),
        ),
        lora=LoraSettings(
            target_modules=tuple(str(item) for item in lora["target_modules"]),
            rank=int(lora["rank"]),
            alpha=int(lora["alpha"]),
            dropout=float(lora["dropout"]),
            bias=str(lora["bias"]),
            use_rslora=bool(lora["use_rslora"]),
            use_dora=bool(lora["use_dora"]),
        ),
        data=PredictionDataSettings(
            variants_per_task=int(data["variants_per_task"]),
            max_sequence_length=int(data["max_sequence_length"]),
            grid_delimiter=str(data["grid_delimiter"]),
            system_header=str(data["system_header"]),
            user_header=str(data["user_header"]),
            assistant_header=str(data["assistant_header"]),
            message_end=str(data["message_end"]),
            pad_to_multiple_of=int(data["pad_to_multiple_of"]),
            synthetic_task_count=int(data["synthetic_task_count"]),
            rearc_task_count=int(data["rearc_task_count"]),
            rearc_examples_per_task=int(data["rearc_examples_per_task"]),
            rearc_pairs_per_variant=int(data["rearc_pairs_per_variant"]),
        ),
        optimization=OptimizationSettings(
            optimizer=str(optimization["optimizer"]),
            scheduler=str(optimization["scheduler"]),
            epochs=int(profile["epochs"]),
            micro_batch_size=int(optimization["micro_batch_size"]),
            gradient_accumulation_steps=int(
                optimization["gradient_accumulation_steps"]
            ),
            validation_batch_size=int(optimization["validation_batch_size"]),
            peak_learning_rate=float(optimization["peak_learning_rate"]),
            end_learning_rate=float(optimization["end_learning_rate"]),
            warmup_ratio=float(optimization["warmup_ratio"]),
            warmup_start_factor=float(optimization["warmup_start_factor"]),
            adam_beta1=float(optimization["adam_beta1"]),
            adam_beta2=float(optimization["adam_beta2"]),
            adam_epsilon=float(optimization["adam_epsilon"]),
            weight_decay=float(optimization["weight_decay"]),
            max_grad_norm=float(optimization["max_grad_norm"]),
        ),
        control=ControlSettings(
            log_every_steps=int(control["log_every_steps"]),
            validate_every_steps=int(control["validate_every_steps"]),
            checkpoint_every_steps=int(control["checkpoint_every_steps"]),
            checkpoint_epochs=tuple(
                int(epoch) for epoch in profile.get("checkpoint_epochs", ())
            ),
            checkpoint_steps=tuple(
                int(step) for step in profile.get("checkpoint_steps", ())
            ),
        ),
        loader=LoaderSettings(
            num_workers=int(loader["num_workers"]),
            pin_memory=bool(loader["pin_memory"]),
            persistent_workers=bool(loader["persistent_workers"]),
        ),
        tracking=TrackingSettings(
            wandb_project=str(tracking["wandb_project"]),
        ),
        profile=PredictionProfileSettings(
            name=profile_name,
            data_source=str(profile["data_source"]),
            guided=bool(profile["guided"]),
            ignore_first_response=bool(profile["ignore_first_response"]),
            initialization=str(profile["initialization"]),
        ),
    )


def load_ttt_training_config(
    path: Path,
    profile_name: str,
    *,
    guidance_budget: Optional[int] = None,
    missing_guidance_policy: Optional[str] = None,
) -> TTTTrainingConfig:
    """Load one online TTT profile and resolve its guidance budget."""
    if not profile_name.strip():
        raise ValueError("profile_name must not be empty.")
    with path.open("r", encoding="utf-8") as file:
        root = _mapping(yaml.safe_load(file), "configuration")

    model = _mapping(root.get("model"), "model")
    lora = _mapping(root.get("lora"), "lora")
    data = _mapping(root.get("data"), "data")
    optimization = _mapping(root.get("optimization"), "optimization")
    loader = _mapping(root.get("loader"), "loader")
    profiles = _mapping(root.get("profiles"), "profiles")
    if profile_name not in profiles:
        available = ", ".join(sorted(str(name) for name in profiles))
        raise ValueError(
            f"unknown TTT profile {profile_name!r}; available profiles: {available}."
        )
    profile = _mapping(profiles[profile_name], f"profiles.{profile_name}")

    resolved_budget = (
        int(profile["default_guidance_budget"])
        if guidance_budget is None
        else guidance_budget
    )
    resolved_missing_policy = (
        str(data["missing_guidance_policy"])
        if missing_guidance_policy is None
        else missing_guidance_policy
    )
    revision_value = model.get("revision")
    revision = str(revision_value) if revision_value is not None else None
    return TTTTrainingConfig(
        schema_version=int(root["schema_version"]),
        seed=int(root["seed"]),
        model=ModelSettings(
            name=str(model["name"]),
            revision=revision,
            quantization_bits=int(model["quantization_bits"]),
            quantization_type=str(model["quantization_type"]),
            double_quantization=bool(model["double_quantization"]),
            dtype=str(model["dtype"]),
            attention_implementation=str(model["attention_implementation"]),
            gradient_checkpointing=bool(model["gradient_checkpointing"]),
        ),
        lora=LoraSettings(
            target_modules=tuple(str(item) for item in lora["target_modules"]),
            rank=int(lora["rank"]),
            alpha=int(lora["alpha"]),
            dropout=float(lora["dropout"]),
            bias=str(lora["bias"]),
            use_rslora=bool(lora["use_rslora"]),
            use_dora=bool(lora["use_dora"]),
        ),
        data=TTTDataSettings(
            variants_per_transform=int(data["variants_per_transform"]),
            max_sequence_length=int(data["max_sequence_length"]),
            grid_delimiter=str(data["grid_delimiter"]),
            system_header=str(data["system_header"]),
            user_header=str(data["user_header"]),
            assistant_header=str(data["assistant_header"]),
            message_end=str(data["message_end"]),
            pad_to_multiple_of=int(data["pad_to_multiple_of"]),
            ignore_first_response=bool(data["ignore_first_response"]),
            empty_guidance_content=str(data["empty_guidance_content"]),
            missing_guidance_policy=resolved_missing_policy,
        ),
        optimization=TTTOptimizationSettings(
            optimizer=str(optimization["optimizer"]),
            scheduler=str(optimization["scheduler"]),
            epochs=int(optimization["epochs"]),
            micro_batch_size=int(optimization["micro_batch_size"]),
            gradient_accumulation_steps=int(
                optimization["gradient_accumulation_steps"]
            ),
            warmup_ratio=float(optimization["warmup_ratio"]),
            warmup_start_learning_rate=float(
                optimization["warmup_start_learning_rate"]
            ),
            min_learning_rate=float(optimization["min_learning_rate"]),
            adam_beta1=float(optimization["adam_beta1"]),
            adam_beta2=float(optimization["adam_beta2"]),
            adam_epsilon=float(optimization["adam_epsilon"]),
            weight_decay=float(optimization["weight_decay"]),
            max_grad_norm=float(optimization["max_grad_norm"]),
        ),
        loader=LoaderSettings(
            num_workers=int(loader["num_workers"]),
            pin_memory=bool(loader["pin_memory"]),
            persistent_workers=bool(loader["persistent_workers"]),
        ),
        profile=TTTProfileSettings(
            name=profile_name,
            guidance_mode=str(profile["guidance_mode"]),
            guidance_budget=resolved_budget,
            peak_learning_rate=float(profile["peak_learning_rate"]),
        ),
    )
