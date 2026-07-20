"""Task-balanced QLoRA training for the Reasoning Model."""

import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import ReasoningTrainingConfig
from .qlora import build_qlora_model_and_optimizer
from .reasoning_data import CACHE_SCHEMA_VERSION


class TaskVariantDataset(Dataset):
    """Expose one distinct variant per task for the selected epoch."""

    def __init__(
        self,
        records_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        seed: int,
    ) -> None:
        if not records_by_task:
            raise ValueError("records_by_task must not be empty.")
        self.records_by_task = {
            task_id: list(records) for task_id, records in records_by_task.items()
        }
        for task_id, records in self.records_by_task.items():
            if not task_id or not records:
                raise ValueError(f"task {task_id!r} has no variants.")
            for record in records:
                if "input_ids" not in record or "assistant_start" not in record:
                    raise ValueError(
                        f"task {task_id!r} contains an incomplete token record."
                    )
        self.task_ids = sorted(self.records_by_task)
        random.Random(seed).shuffle(self.task_ids)
        self.epoch = 0

    @property
    def minimum_variant_count(self) -> int:
        return min(len(records) for records in self.records_by_task.values())

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer.")
        if epoch >= self.minimum_variant_count:
            raise ValueError(
                f"epoch {epoch} requires at least {epoch + 1} variants per task."
            )
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.task_ids)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        task_id = self.task_ids[index]
        return self.records_by_task[task_id][self.epoch]


class AssistantOnlyCollator:
    """Right-pad token records and mask all non-assistant labels."""

    def __init__(self, *, pad_token_id: int, pad_to_multiple_of: int = 8) -> None:
        if isinstance(pad_token_id, bool) or pad_token_id < 0:
            raise ValueError("pad_token_id must be a non-negative integer.")
        if isinstance(pad_to_multiple_of, bool) or pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be a positive integer.")
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(
        self, features: Sequence[Mapping[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("cannot collate an empty feature sequence.")
        sequences: List[torch.Tensor] = []
        assistant_starts: List[int] = []
        for index, feature in enumerate(features):
            input_ids = feature.get("input_ids")
            if isinstance(input_ids, torch.Tensor):
                sequence = input_ids.to(dtype=torch.long)
            elif isinstance(input_ids, list) and all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in input_ids
            ):
                sequence = torch.tensor(input_ids, dtype=torch.long)
            else:
                raise TypeError(f"features[{index}].input_ids must be integer tokens.")
            if sequence.ndim != 1 or sequence.numel() == 0:
                raise ValueError(
                    f"features[{index}].input_ids must be one-dimensional."
                )
            assistant_start = feature.get("assistant_start")
            if (
                not isinstance(assistant_start, int)
                or isinstance(assistant_start, bool)
                or not 0 <= assistant_start < sequence.numel()
            ):
                raise ValueError(f"features[{index}] has an invalid assistant_start.")
            sequences.append(sequence)
            assistant_starts.append(assistant_start)

        longest = max(sequence.numel() for sequence in sequences)
        padded_length = (
            math.ceil(longest / self.pad_to_multiple_of) * self.pad_to_multiple_of
        )
        batch_size = len(sequences)
        input_ids = torch.full(
            (batch_size, padded_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, padded_length), dtype=torch.long)
        labels = torch.full((batch_size, padded_length), -100, dtype=torch.long)
        for index, (sequence, assistant_start) in enumerate(
            zip(sequences, assistant_starts)
        ):
            length = sequence.numel()
            input_ids[index, :length] = sequence
            attention_mask[index, :length] = 1
            labels[index, assistant_start:length] = sequence[assistant_start:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def optimizer_steps_per_epoch(
    task_count: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Return optimizer updates per epoch, including a final partial update."""
    for value, name in (
        (task_count, "task_count"),
        (micro_batch_size, "micro_batch_size"),
        (gradient_accumulation_steps, "gradient_accumulation_steps"),
    ):
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    micro_batches = math.ceil(task_count / micro_batch_size)
    return math.ceil(micro_batches / gradient_accumulation_steps)


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
    warmup_start_factor: float,
    peak_learning_rate: float,
    end_learning_rate: float,
) -> LRScheduler:
    """Build the same LinearLR-to-CosineAnnealingLR chain as the legacy run."""
    warmup_steps = int(total_steps * warmup_ratio)
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup steps must be smaller than total steps.")
    if optimizer.param_groups[0]["lr"] != peak_learning_rate:
        raise ValueError("optimizer learning rate must equal peak_learning_rate.")
    cosine_steps = max(1, total_steps - warmup_steps)
    if warmup_steps == 0:
        return CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=end_learning_rate,
        )
    warmup = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=end_learning_rate,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def load_reasoning_cache(path: Path) -> Mapping[str, Any]:
    """Load a trusted cache produced by the repository's cache builder."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Reasoning Model cache must contain a mapping.")
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported Reasoning Model cache schema version.")
    splits = payload.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("Reasoning Model cache has no split mapping.")
    for split in ("train", "validation", "validation_augmented"):
        if not isinstance(splits.get(split), Mapping) or not splits[split]:
            raise ValueError(f"Reasoning Model cache split {split!r} is empty.")
    return payload


def validate_reasoning_cache_config(
    cache: Mapping[str, Any],
    config: ReasoningTrainingConfig,
) -> None:
    """Reject caches built with settings that affect tokenized data."""
    metadata = cache.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Reasoning Model cache has no metadata.")
    cached_config = metadata.get("config")
    if not isinstance(cached_config, Mapping):
        raise ValueError("Reasoning Model cache has no resolved configuration.")
    expected = config.to_dict()
    if cached_config.get("seed") != expected["seed"]:
        raise ValueError("Reasoning Model cache was built with a different seed.")
    if cached_config.get("data") != expected["data"]:
        raise ValueError("Reasoning Model cache data settings do not match the run.")
    cached_model = cached_config.get("model")
    if not isinstance(cached_model, Mapping):
        raise ValueError("Reasoning Model cache has no model settings.")
    for field in ("name", "revision"):
        if cached_model.get(field) != expected["model"][field]:
            raise ValueError(
                f"Reasoning Model cache model {field} does not match the run."
            )


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def build_model_and_optimizer(
    config: ReasoningTrainingConfig,
) -> tuple[Any, torch.optim.Optimizer]:
    """Load the quantized base model, attach LoRA, and create AdamW 8-bit."""
    return build_qlora_model_and_optimizer(
        config,
        workload_name="Reasoning Model",
    )


def _autocast(device: torch.device, dtype: torch.dtype) -> Any:
    if device.type == "cuda" and dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def evaluate_model(
    model: Any,
    loader: DataLoader,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """Compute token-weighted assistant loss for one validation view."""
    model.eval()
    weighted_loss = 0.0
    target_tokens = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.to(device, non_blocking=True) for key, value in batch.items()
            }
            with _autocast(device, dtype):
                output = model(**batch)
            tokens = int((batch["labels"] != -100).sum().item())
            weighted_loss += float(output.loss.detach().item()) * tokens
            target_tokens += tokens
    model.train()
    if target_tokens == 0:
        raise ValueError("validation loader contains no assistant target tokens.")
    return weighted_loss / target_tokens


def _save_checkpoint(
    model: Any,
    directory: Path,
    *,
    global_step: int,
    current_epoch: int,
    completed_epochs: int,
    reasons: Sequence[str],
    validation_metrics: Optional[Mapping[str, float]],
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(directory, safe_serialization=True)
    state: Dict[str, Any] = {
        "global_step": global_step,
        "current_epoch": current_epoch,
        "completed_epochs": completed_epochs,
        "reasons": list(reasons),
    }
    if validation_metrics is not None:
        state["validation"] = dict(validation_metrics)
    with (directory / "training_state.json").open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, sort_keys=True)
        file.write("\n")


def _model_device(model: Any) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("could not determine the model device.")


def train_reasoning_model(
    *,
    config: ReasoningTrainingConfig,
    cache_path: Path,
    output_directory: Path,
    run_name: str,
    use_wandb: bool,
) -> Mapping[str, Any]:
    """Run the final thesis-aligned Reasoning Model training protocol."""
    if not run_name.strip():
        raise ValueError("run_name must not be empty.")
    run_directory = output_directory / run_name
    if run_directory.exists():
        raise FileExistsError(f"training run already exists: {run_directory}.")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    cache = load_reasoning_cache(cache_path)
    validate_reasoning_cache_config(cache, config)
    splits = cache["splits"]
    train_dataset = TaskVariantDataset(splits["train"], seed=config.seed)
    if train_dataset.minimum_variant_count < config.optimization.epochs:
        raise ValueError(
            "training cache does not contain one distinct variant per task and epoch."
        )
    validation_dataset = TaskVariantDataset(splits["validation"], seed=config.seed)
    augmented_validation_dataset = TaskVariantDataset(
        splits["validation_augmented"],
        seed=config.seed,
    )

    metadata = cache.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Reasoning Model cache has no metadata.")
    tokenizer_metadata = metadata.get("tokenizer")
    if not isinstance(tokenizer_metadata, Mapping):
        raise ValueError("Reasoning Model cache has no tokenizer metadata.")
    pad_token_id = tokenizer_metadata.get("pad_token_id")
    if not isinstance(pad_token_id, int):
        raise ValueError("Reasoning Model cache has no integer pad token ID.")
    collator = AssistantOnlyCollator(
        pad_token_id=pad_token_id,
        pad_to_multiple_of=config.data.pad_to_multiple_of,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.optimization.micro_batch_size,
        shuffle=False,
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        persistent_workers=config.loader.persistent_workers,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.optimization.validation_batch_size,
        shuffle=False,
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        persistent_workers=config.loader.persistent_workers,
        collate_fn=collator,
    )
    augmented_validation_loader = DataLoader(
        augmented_validation_dataset,
        batch_size=config.optimization.validation_batch_size,
        shuffle=False,
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        persistent_workers=config.loader.persistent_workers,
        collate_fn=collator,
    )

    steps_per_epoch = optimizer_steps_per_epoch(
        len(train_dataset),
        config.optimization.micro_batch_size,
        config.optimization.gradient_accumulation_steps,
    )
    total_steps = config.optimization.epochs * steps_per_epoch
    model, optimizer = build_model_and_optimizer(config)
    scheduler = build_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=config.optimization.warmup_ratio,
        warmup_start_factor=config.optimization.warmup_start_factor,
        peak_learning_rate=config.optimization.peak_learning_rate,
        end_learning_rate=config.optimization.end_learning_rate,
    )
    device = _model_device(model)
    dtype = _torch_dtype(config.model.dtype)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    run_directory.mkdir(parents=True, exist_ok=False)
    with (run_directory / "resolved_config.json").open("w", encoding="utf-8") as file:
        json.dump(config.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")

    wandb_run: Any = None
    if use_wandb:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("install the optional tracking dependencies.") from error
        wandb_run = wandb.init(
            project=config.tracking.wandb_project,
            name=run_name,
            config=config.to_dict(),
        )

    global_step = 0
    saved_checkpoints: Set[str] = set()
    last_validation_step: Optional[int] = None
    last_validation: Optional[Dict[str, float]] = None
    optimizer.zero_grad(set_to_none=True)

    def run_validation() -> Dict[str, float]:
        nonlocal last_validation_step, last_validation
        metrics = {
            "val/loss": evaluate_model(
                model,
                validation_loader,
                device=device,
                dtype=dtype,
            ),
            "val/loss_aug": evaluate_model(
                model,
                augmented_validation_loader,
                device=device,
                dtype=dtype,
            ),
        }
        last_validation_step = global_step
        last_validation = metrics
        if wandb_run is not None:
            wandb_run.log(metrics, step=global_step)
        return metrics

    def save_checkpoint(
        current_epoch: int,
        completed_epochs: int,
        reasons: Sequence[str],
        *,
        directory_name: Optional[str] = None,
    ) -> None:
        checkpoint_name = directory_name or f"ckpt_step_{global_step}"
        if checkpoint_name in saved_checkpoints:
            return
        metrics = (
            last_validation if last_validation_step == global_step else run_validation()
        )
        _save_checkpoint(
            model,
            run_directory / checkpoint_name,
            global_step=global_step,
            current_epoch=current_epoch,
            completed_epochs=completed_epochs,
            reasons=reasons,
            validation_metrics=metrics,
        )
        saved_checkpoints.add(checkpoint_name)

    loss_window = 0.0
    tokens_window = 0
    try:
        # Preserve the update order of the original research training loop.
        for epoch_index in range(config.optimization.epochs):
            completed_epoch = epoch_index + 1
            train_dataset.set_epoch(epoch_index)
            steps_in_epoch = 0
            micro_steps = 0
            loss_step = 0.0
            tokens_step = 0
            for batch_index, batch in enumerate(
                tqdm(train_loader, desc=f"epoch {completed_epoch}"),
                start=1,
            ):
                batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch.items()
                }
                micro_steps += 1
                with _autocast(device, dtype):
                    output = model(**batch)
                    loss = output.loss
                target_tokens = int((batch["labels"] != -100).sum().item())
                loss_step += float(loss.detach().item()) * target_tokens
                tokens_step += target_tokens
                (loss / config.optimization.gradient_accumulation_steps).backward()

                is_accumulation_boundary = (
                    micro_steps % config.optimization.gradient_accumulation_steps == 0
                )
                is_last_batch = batch_index == len(train_loader)
                if not is_accumulation_boundary and not is_last_batch:
                    continue
                remainder = (
                    micro_steps % config.optimization.gradient_accumulation_steps
                )
                if is_last_batch and remainder:
                    correction = (
                        config.optimization.gradient_accumulation_steps / remainder
                    )
                    for parameter in trainable_parameters:
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    config.optimization.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                steps_in_epoch += 1

                average_loss_step = loss_step / max(tokens_step, 1)
                loss_window += loss_step
                tokens_window += tokens_step
                tokens_in_batch = tokens_step
                loss_step = 0.0
                tokens_step = 0

                if global_step % config.control.log_every_steps == 0:
                    metrics = {
                        "train/loss": loss_window / max(tokens_window, 1),
                        "train/loss_step": average_loss_step,
                        "train/tokens_in_batch": tokens_in_batch,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/grad_norm": float(grad_norm.item()),
                        "train/local_steps": steps_in_epoch,
                        "train/epoch": (epoch_index + steps_in_epoch / steps_per_epoch),
                    }
                    if wandb_run is not None:
                        wandb_run.log(metrics, step=global_step)
                    loss_window = 0.0
                    tokens_window = 0

                if global_step % config.control.validate_every_steps == 0:
                    run_validation()
                if global_step % config.control.checkpoint_every_steps == 0:
                    save_checkpoint(
                        completed_epoch,
                        epoch_index,
                        ["periodic_step"],
                    )

            checkpoint_reasons: List[str] = []
            if completed_epoch in config.control.checkpoint_epochs:
                checkpoint_reasons.append("configured_epoch")
            if checkpoint_reasons:
                save_checkpoint(
                    completed_epoch,
                    completed_epoch,
                    checkpoint_reasons,
                )

        save_checkpoint(
            config.optimization.epochs,
            config.optimization.epochs,
            ["final_epoch"],
            directory_name=f"end_epoch_{config.optimization.epochs - 1}",
        )

        summary = {
            "run_name": run_name,
            "global_steps": global_step,
            "completed_epochs": config.optimization.epochs,
            "steps_per_epoch": steps_per_epoch,
            "saved_checkpoints": sorted(saved_checkpoints),
            "final_validation": last_validation,
        }
        with (run_directory / "training_summary.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(summary, file, indent=2, sort_keys=True)
            file.write("\n")
        return summary
    finally:
        if wandb_run is not None:
            wandb_run.finish()
