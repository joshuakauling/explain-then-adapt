"""Shared QLoRA construction for offline training and online adaptation."""

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Protocol

import torch

from .config import LoraSettings, ModelSettings


class QuantizedTrainingConfig(Protocol):
    @property
    def model(self) -> ModelSettings: ...

    @property
    def lora(self) -> LoraSettings: ...


class OptimizerSettings(Protocol):
    @property
    def adam_beta1(self) -> float: ...

    @property
    def adam_beta2(self) -> float: ...

    @property
    def adam_epsilon(self) -> float: ...

    @property
    def weight_decay(self) -> float: ...


class OfflineOptimizerSettings(OptimizerSettings, Protocol):
    @property
    def peak_learning_rate(self) -> float: ...


class OfflineTrainingConfig(QuantizedTrainingConfig, Protocol):

    @property
    def optimization(self) -> OfflineOptimizerSettings: ...


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_quantized_training_base(
    config: QuantizedTrainingConfig,
    *,
    workload_name: str,
    base_model_path: Optional[Path] = None,
) -> Any:
    """Load and prepare one quantized causal LM for adapter training."""
    if base_model_path is not None and not base_model_path.is_dir():
        raise FileNotFoundError(
            f"base model directory does not exist: {base_model_path}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(f"{workload_name} QLoRA training requires a CUDA GPU.")
    try:
        from peft.utils import (  # type: ignore[import-not-found]
            prepare_model_for_kbit_training,
        )
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            "install the optional training dependencies with "
            "`python -m pip install -e '.[training]'`."
        ) from error

    dtype = _torch_dtype(config.model.dtype)
    quantization_kwargs: Dict[str, Any]
    if config.model.quantization_bits == 4:
        quantization_kwargs = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": config.model.quantization_type,
            "bnb_4bit_use_double_quant": config.model.double_quantization,
            "bnb_4bit_compute_dtype": dtype,
        }
    else:
        quantization_kwargs = {"load_in_8bit": True}
    quantization = BitsAndBytesConfig(**quantization_kwargs)
    model_kwargs: Dict[str, Any] = {
        "quantization_config": quantization,
        "dtype": dtype,
        "attn_implementation": config.model.attention_implementation,
        "device_map": {"": torch.cuda.current_device()},
    }
    if config.model.revision is not None and base_model_path is None:
        model_kwargs["revision"] = config.model.revision
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path) if base_model_path is not None else config.model.name,
        **model_kwargs,
    )
    model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=config.model.gradient_checkpointing,
    )
    model.config.use_cache = False
    return model


def build_lora_config(settings: LoraSettings) -> Any:
    """Create the shared PEFT LoRA configuration lazily."""
    try:
        from peft import LoraConfig, TaskType  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "install the optional training dependencies with "
            "`python -m pip install -e '.[training]'`."
        ) from error

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(settings.target_modules),
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        bias=settings.bias,
        use_rslora=settings.use_rslora,
        use_dora=settings.use_dora,
    )


def build_adamw_8bit_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    settings: OptimizerSettings,
    *,
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Create the final AdamW 8-bit optimizer for trainable parameters."""
    try:
        from bitsandbytes.optim import AdamW8bit  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "install the optional training dependencies with "
            "`python -m pip install -e '.[training]'`."
        ) from error

    trainable_parameters = list(parameters)
    if not trainable_parameters:
        raise RuntimeError("QLoRA model exposes no trainable parameters.")
    return AdamW8bit(
        trainable_parameters,
        lr=learning_rate,
        betas=(settings.adam_beta1, settings.adam_beta2),
        eps=settings.adam_epsilon,
        weight_decay=settings.weight_decay,
    )


def build_qlora_model_and_optimizer(
    config: OfflineTrainingConfig,
    *,
    workload_name: str,
    base_model_path: Optional[Path] = None,
) -> tuple[Any, torch.optim.Optimizer]:
    """Load a quantized base, attach fresh LoRA, and create AdamW 8-bit."""
    try:
        from peft import get_peft_model  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "install the optional training dependencies with "
            "`python -m pip install -e '.[training]'`."
        ) from error

    base_model = load_quantized_training_base(
        config,
        workload_name=workload_name,
        base_model_path=base_model_path,
    )
    model = get_peft_model(base_model, build_lora_config(config.lora))

    model.print_trainable_parameters()
    model.train()
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = build_adamw_8bit_optimizer(
        trainable_parameters,
        config.optimization,
        learning_rate=config.optimization.peak_learning_rate,
    )
    return model, optimizer
