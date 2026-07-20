"""Task-balanced QLoRA training for the Prediction Model."""

import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import PredictionTrainingConfig
from .prediction_data import (
    PREDICTION_CACHE_SCHEMA_VERSION,
    prediction_cache_contract,
)
from .qlora import build_qlora_model_and_optimizer
from .reasoning_trainer import (
    build_cosine_scheduler,
    evaluate_model,
    optimizer_steps_per_epoch,
)


class PredictionTaskVariantDataset(Dataset):
    """Expose one distinct Prediction Model variant per task and epoch."""

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
                if "input_ids" not in record or "assistant_spans" not in record:
                    raise ValueError(
                        f"task {task_id!r} contains an incomplete PM token record."
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


class PredictionAssistantCollator:
    """Right-pad records and target only selected assistant grid spans."""

    def __init__(
        self,
        *,
        pad_token_id: int,
        ignore_first_response: bool,
        pad_to_multiple_of: int = 8,
    ) -> None:
        if isinstance(pad_token_id, bool) or pad_token_id < 0:
            raise ValueError("pad_token_id must be a non-negative integer.")
        if isinstance(pad_to_multiple_of, bool) or pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be a positive integer.")
        self.pad_token_id = pad_token_id
        self.ignore_first_response = ignore_first_response
        self.pad_to_multiple_of = pad_to_multiple_of

    @staticmethod
    def _sequence(feature: Mapping[str, Any], index: int) -> torch.Tensor:
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
            raise ValueError(f"features[{index}].input_ids must be one-dimensional.")
        return sequence

    @staticmethod
    def _spans(
        feature: Mapping[str, Any],
        *,
        index: int,
        sequence_length: int,
    ) -> List[tuple[int, int]]:
        value = feature.get("assistant_spans")
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"features[{index}] has no assistant spans.")
        spans: List[tuple[int, int]] = []
        previous_end = 0
        for span_index, span in enumerate(value):
            if (
                not isinstance(span, (list, tuple))
                or len(span) != 2
                or any(
                    not isinstance(position, int) or isinstance(position, bool)
                    for position in span
                )
            ):
                raise ValueError(
                    f"features[{index}].assistant_spans[{span_index}] is invalid."
                )
            start, end = int(span[0]), int(span[1])
            if not previous_end <= start < end <= sequence_length:
                raise ValueError(
                    f"features[{index}].assistant_spans[{span_index}] is out of order."
                )
            spans.append((start, end))
            previous_end = end
        return spans

    def __call__(
        self, features: Sequence[Mapping[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("cannot collate an empty feature sequence.")
        sequences: List[torch.Tensor] = []
        selected_spans: List[List[tuple[int, int]]] = []
        for index, feature in enumerate(features):
            sequence = self._sequence(feature, index)
            spans = self._spans(
                feature,
                index=index,
                sequence_length=sequence.numel(),
            )
            if self.ignore_first_response:
                spans = spans[1:]
            if not spans:
                raise ValueError(
                    f"features[{index}] has no target after response masking."
                )
            sequences.append(sequence)
            selected_spans.append(spans)

        longest = max(sequence.numel() for sequence in sequences)
        padded_length = (
            math.ceil(longest / self.pad_to_multiple_of) * self.pad_to_multiple_of
        )
        input_ids = torch.full(
            (len(sequences), padded_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        for index, (sequence, spans) in enumerate(zip(sequences, selected_spans)):
            length = sequence.numel()
            input_ids[index, :length] = sequence
            attention_mask[index, :length] = 1
            for start, end in spans:
                labels[index, start:end] = sequence[start:end]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_prediction_cache(path: Path) -> Mapping[str, Any]:
    """Load a trusted cache produced by the PM cache builder."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Prediction Model cache must contain a mapping.")
    if (
        payload.get("schema_version") != PREDICTION_CACHE_SCHEMA_VERSION
        or payload.get("kind") != "prediction_model"
    ):
        raise ValueError("unsupported Prediction Model cache schema.")
    splits = payload.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("Prediction Model cache has no split mapping.")
    for split in ("train", "validation", "validation_augmented"):
        if not isinstance(splits.get(split), Mapping) or not splits[split]:
            raise ValueError(f"Prediction Model cache split {split!r} is empty.")
    return payload


def validate_prediction_cache_config(
    cache: Mapping[str, Any],
    config: PredictionTrainingConfig,
) -> None:
    """Reject a cache whose tokenizer-facing contract differs from the run."""
    metadata = cache.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Prediction Model cache has no metadata.")
    if metadata.get("cache_contract") != prediction_cache_contract(config):
        raise ValueError("Prediction Model cache settings do not match the run.")


def build_prediction_model_and_optimizer(
    config: PredictionTrainingConfig,
    *,
    initial_model_path: Optional[Path],
) -> tuple[Any, torch.optim.Optimizer]:
    """Attach a fresh PM adapter to the configured or merged base model."""
    return build_qlora_model_and_optimizer(
        config,
        workload_name="Prediction Model",
        base_model_path=initial_model_path,
    )


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _autocast(device: torch.device, dtype: torch.dtype) -> Any:
    if device.type == "cuda" and dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _model_device(model: Any) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("could not determine the model device.")


def _save_checkpoint(
    model: Any,
    directory: Path,
    *,
    global_step: int,
    current_epoch: int,
    completed_epochs: int,
    profile: str,
    reasons: Sequence[str],
    validation_metrics: Optional[Mapping[str, float]],
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(directory, safe_serialization=True)
    state: Dict[str, Any] = {
        "global_step": global_step,
        "current_epoch": current_epoch,
        "completed_epochs": completed_epochs,
        "profile": profile,
        "reasons": list(reasons),
    }
    if validation_metrics is not None:
        state["validation"] = dict(validation_metrics)
    with (directory / "training_state.json").open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, sort_keys=True)
        file.write("\n")


def train_prediction_model(
    *,
    config: PredictionTrainingConfig,
    cache_path: Path,
    output_directory: Path,
    run_name: str,
    use_wandb: bool,
    initial_model_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    """Run one final thesis-aligned Prediction Model training profile."""
    if not run_name.strip():
        raise ValueError("run_name must not be empty.")
    if config.profile.initialization == "merged_model" and initial_model_path is None:
        raise ValueError(
            f"profile {config.profile.name!r} requires --initial-model containing "
            "the merged Unguided-ReARC model."
        )
    if config.profile.initialization == "pretrained" and initial_model_path is not None:
        raise ValueError(
            f"profile {config.profile.name!r} starts from the pretrained model and "
            "must not receive an initial model override."
        )
    run_directory = output_directory / run_name
    if run_directory.exists():
        raise FileExistsError(f"training run already exists: {run_directory}.")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    cache = load_prediction_cache(cache_path)
    validate_prediction_cache_config(cache, config)
    splits = cache["splits"]
    train_dataset = PredictionTaskVariantDataset(
        splits["train"],
        seed=config.seed,
    )
    if len(train_dataset) != config.expected_task_count:
        raise ValueError(
            f"training cache contains {len(train_dataset)} tasks; expected "
            f"{config.expected_task_count}."
        )
    if train_dataset.minimum_variant_count < config.optimization.epochs:
        raise ValueError(
            "training cache does not contain one distinct variant per task and epoch."
        )
    validation_dataset = PredictionTaskVariantDataset(
        splits["validation"],
        seed=config.seed,
    )
    augmented_validation_dataset = PredictionTaskVariantDataset(
        splits["validation_augmented"],
        seed=config.seed,
    )

    metadata = cache.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Prediction Model cache has no metadata.")
    tokenizer_metadata = metadata.get("tokenizer")
    if not isinstance(tokenizer_metadata, Mapping):
        raise ValueError("Prediction Model cache has no tokenizer metadata.")
    pad_token_id = tokenizer_metadata.get("pad_token_id")
    if not isinstance(pad_token_id, int):
        raise ValueError("Prediction Model cache has no integer pad token ID.")
    collator = PredictionAssistantCollator(
        pad_token_id=pad_token_id,
        ignore_first_response=config.profile.ignore_first_response,
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
    model, optimizer = build_prediction_model_and_optimizer(
        config,
        initial_model_path=initial_model_path,
    )
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
    resolved_config = config.to_dict()
    resolved_config["initial_model_path"] = (
        str(initial_model_path) if initial_model_path is not None else None
    )
    with (run_directory / "resolved_config.json").open("w", encoding="utf-8") as file:
        json.dump(resolved_config, file, indent=2, sort_keys=True)
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
            config=resolved_config,
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
            profile=config.profile.name,
            reasons=reasons,
            validation_metrics=metrics,
        )
        saved_checkpoints.add(checkpoint_name)

    loss_window = 0.0
    tokens_window = 0
    try:
        # Keep the update order of the original PM training loop.
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
                        "train/epoch": epoch_index + steps_in_epoch / steps_per_epoch,
                    }
                    if wandb_run is not None:
                        wandb_run.log(metrics, step=global_step)
                    loss_window = 0.0
                    tokens_window = 0

                if global_step % config.control.validate_every_steps == 0:
                    run_validation()
                checkpoint_reasons: List[str] = []
                if global_step % config.control.checkpoint_every_steps == 0:
                    checkpoint_reasons.append("periodic_step")
                if global_step in config.control.checkpoint_steps:
                    checkpoint_reasons.append("configured_step")
                if checkpoint_reasons:
                    save_checkpoint(
                        completed_epoch,
                        epoch_index,
                        checkpoint_reasons,
                    )

            if completed_epoch in config.control.checkpoint_epochs:
                save_checkpoint(
                    completed_epoch,
                    completed_epoch,
                    ["configured_epoch"],
                )

        save_checkpoint(
            config.optimization.epochs,
            config.optimization.epochs,
            ["final_epoch"],
            directory_name=f"end_epoch_{config.optimization.epochs - 1}",
        )

        summary = {
            "run_name": run_name,
            "profile": config.profile.name,
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
