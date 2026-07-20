"""Per-task QLoRA adaptation for the final online TTT protocol."""

import gc
import hashlib
import json
import math
import random
import shutil
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from explain_then_adapt.arc.types import TRANSFORM_CODES

from .config import TTTTrainingConfig
from .prediction_trainer import PredictionAssistantCollator
from .qlora import (
    build_adamw_8bit_optimizer,
    build_lora_config,
    load_quantized_training_base,
)
from .ttt_data import (
    TTTAugmentation,
    build_ttt_records,
    generate_ttt_augmentation_plan,
    load_ttt_augmentation_plan,
    load_ttt_guidance,
    normalize_ttt_task_ids,
    ttt_augmentation_plan_payload,
)

TTT_BOOTSTRAP_ADAPTER = "ttt_bootstrap"


class TTTRecordDataset(Dataset):
    """Small ordered dataset for the 64 variants of one task."""

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.records[index]


def _stable_seed(seed: int, *parts: str) -> int:
    material = ":".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    raise RuntimeError("could not determine the TTT model device.")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary JSON file already exists: {temporary}.")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(path)


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON file must contain a mapping: {path}.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_source_sha256(tasks_directory: Path, task_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for task_id in sorted(task_ids):
        path = tasks_directory / f"{task_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"ARC task file does not exist: {path}.")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def load_ttt_tokenizer(prediction_model_path: Path) -> Any:
    """Load the tokenizer stored beside the merged Prediction Model."""
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "install the optional training dependencies with "
            "`python -m pip install -e '.[training]'`."
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(
        str(prediction_model_path),
        use_fast=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("TTT tokenizer has neither a pad nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_ttt_model(
    config: TTTTrainingConfig,
    *,
    prediction_model_path: Path,
) -> Tuple[Any, Any]:
    """Load the merged PM once and attach an unused bootstrap adapter."""
    try:
        from peft import get_peft_model  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "install the optional training dependencies with "
            "`python -m pip install -e '.[training]'`."
        ) from error

    base_model = load_quantized_training_base(
        config,
        workload_name="Test-Time Training",
        base_model_path=prediction_model_path,
    )
    lora_config = build_lora_config(config.lora)
    model = get_peft_model(
        base_model,
        lora_config,
        adapter_name=TTT_BOOTSTRAP_ADAPTER,
    )
    model.config.use_cache = False
    model.train()
    model.print_trainable_parameters()
    return model, lora_config


def build_ttt_optimizer(
    model: Any,
    config: TTTTrainingConfig,
) -> torch.optim.Optimizer:
    """Create a new optimizer over only the currently active task adapter."""
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if any(TTT_BOOTSTRAP_ADAPTER in name for name, _ in named_parameters):
        raise RuntimeError("the inactive TTT bootstrap adapter is still trainable.")
    return build_adamw_8bit_optimizer(
        (parameter for _, parameter in named_parameters),
        config.optimization,
        learning_rate=config.profile.peak_learning_rate,
    )


def build_ttt_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
    warmup_start_learning_rate: float,
    peak_learning_rate: float,
    min_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warm linearly from zero, then decay by cosine to the thesis LR floor."""
    if isinstance(total_steps, bool) or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer.")
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1).")
    if not 0 <= warmup_start_learning_rate <= peak_learning_rate:
        raise ValueError("invalid warmup start learning rate.")
    if not 0 < min_learning_rate <= peak_learning_rate:
        raise ValueError("invalid minimum learning rate.")

    for group in optimizer.param_groups:
        group["lr"] = peak_learning_rate
    warmup_steps = int(total_steps * warmup_ratio)
    decay_steps = total_steps - warmup_steps
    start_factor = warmup_start_learning_rate / peak_learning_rate
    end_factor = min_learning_rate / peak_learning_rate

    def lr_lambda(current_step: int) -> float:
        if warmup_steps and current_step <= warmup_steps:
            progress = current_step / warmup_steps
            return start_factor + progress * (1.0 - start_factor)
        if decay_steps <= 0:
            return end_factor
        progress = min(
            1.0,
            max(0.0, (current_step - warmup_steps) / decay_steps),
        )
        return end_factor + 0.5 * (1.0 - end_factor) * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _tokenizer_metadata(tokenizer: Any) -> Dict[str, Any]:
    return {
        "class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "padding_side": tokenizer.padding_side,
    }


def _adapter_files_complete(
    directory: Path,
    *,
    expected_task_id: Optional[str] = None,
) -> bool:
    if not directory.is_dir():
        return False
    manifest_path = directory / "ttt_manifest.json"
    if not manifest_path.is_file() or not (directory / "adapter_config.json").is_file():
        return False
    weight_names = (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "adapter_model.pt",
    )
    if not any(
        (directory / name).is_file() and (directory / name).stat().st_size > 0
        for name in weight_names
    ):
        return False
    manifest = _load_json_mapping(manifest_path)
    expected_id = expected_task_id or directory.name
    steps = manifest.get("steps")
    training_order = manifest.get("training_order")
    transform_counts = manifest.get("transform_counts")
    guidance_counts = manifest.get("guidance_counts")
    step_ids = (
        [step.get("variant_id") for step in steps]
        if isinstance(steps, list) and all(isinstance(step, Mapping) for step in steps)
        else []
    )
    valid_guidance_counts = (
        isinstance(guidance_counts, Mapping)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in guidance_counts.values()
        )
        and sum(guidance_counts.values()) == 64
    )
    return (
        manifest.get("schema_version") == 1
        and manifest.get("kind") == "ttt_task_adapter"
        and manifest.get("task_id") == expected_id
        and manifest.get("optimizer_updates") == 64
        and isinstance(steps, list)
        and len(steps) == 64
        and [step.get("step") for step in steps if isinstance(step, Mapping)]
        == list(range(1, 65))
        and isinstance(training_order, list)
        and len(training_order) == 64
        and step_ids == training_order
        and all(isinstance(step_id, str) for step_id in step_ids)
        and len(set(step_ids)) == 64
        and isinstance(transform_counts, Mapping)
        and dict(transform_counts) == {code: 8 for code in TRANSFORM_CODES}
        and valid_guidance_counts
    )


def _save_task_adapter(
    *,
    model: Any,
    run_directory: Path,
    task_id: str,
    manifest: Mapping[str, Any],
) -> Path:
    final_directory = run_directory / task_id
    temporary_root = run_directory / f".{task_id}.partial"
    if final_directory.exists():
        raise FileExistsError(
            f"TTT adapter directory already exists: {final_directory}."
        )
    if temporary_root.exists():
        raise FileExistsError(
            f"incomplete TTT adapter directory already exists: {temporary_root}."
        )

    model.save_pretrained(
        temporary_root,
        selected_adapters=[task_id],
        safe_serialization=True,
    )
    nested_directory = temporary_root / task_id
    adapter_directory = (
        nested_directory
        if (nested_directory / "adapter_config.json").is_file()
        else temporary_root
    )
    if not (adapter_directory / "adapter_config.json").is_file():
        raise RuntimeError(f"PEFT did not save adapter_config.json for {task_id!r}.")
    _atomic_write_json(adapter_directory / "ttt_manifest.json", manifest)
    if not _adapter_files_complete(
        adapter_directory,
        expected_task_id=task_id,
    ):
        raise RuntimeError(f"saved TTT adapter for {task_id!r} is incomplete.")

    adapter_directory.rename(final_directory)
    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    return final_directory


def _task_manifest(
    *,
    task_id: str,
    config: TTTTrainingConfig,
    task_seed: int,
    records: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    guidance_counts = Counter(str(record["guidance_status"]) for record in records)
    transform_counts = Counter(str(record["transformation_code"]) for record in records)
    return {
        "schema_version": 1,
        "kind": "ttt_task_adapter",
        "task_id": task_id,
        "profile": config.profile.name,
        "guidance_mode": config.profile.guidance_mode,
        "guidance_budget": config.profile.guidance_budget,
        "task_seed": task_seed,
        "optimizer_updates": len(steps),
        "processed_tokens": sum(int(record["sequence_length"]) for record in records),
        "target_tokens": sum(int(record["target_token_count"]) for record in records),
        "guidance_counts": dict(sorted(guidance_counts.items())),
        "transform_counts": dict(sorted(transform_counts.items())),
        "training_order": [str(step["variant_id"]) for step in steps],
        "steps": list(steps),
    }


def _completed_manifests(
    run_directory: Path,
    task_ids: Sequence[str],
) -> Dict[str, Mapping[str, Any]]:
    completed: Dict[str, Mapping[str, Any]] = {}
    for task_id in task_ids:
        task_directory = run_directory / task_id
        partial_directory = run_directory / f".{task_id}.partial"
        if partial_directory.exists():
            raise RuntimeError(
                f"incomplete task output requires inspection before resume: "
                f"{partial_directory}."
            )
        if not task_directory.exists():
            continue
        if not _adapter_files_complete(task_directory):
            raise RuntimeError(
                f"task output exists but is incomplete: {task_directory}."
            )
        completed[task_id] = _load_json_mapping(task_directory / "ttt_manifest.json")
    return completed


def _run_summary(
    *,
    config: TTTTrainingConfig,
    task_ids: Sequence[str],
    manifests: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "ttt_run",
        "profile": config.profile.name,
        "guidance_mode": config.profile.guidance_mode,
        "guidance_budget": config.profile.guidance_budget,
        "task_count": len(task_ids),
        "completed_task_count": len(manifests),
        "completed_tasks": sorted(manifests),
        "optimizer_updates": sum(
            int(manifest["optimizer_updates"]) for manifest in manifests.values()
        ),
        "processed_tokens": sum(
            int(manifest["processed_tokens"]) for manifest in manifests.values()
        ),
        "target_tokens": sum(
            int(manifest["target_tokens"]) for manifest in manifests.values()
        ),
    }


def _train_one_task(
    *,
    model: Any,
    lora_config: Any,
    config: TTTTrainingConfig,
    task_id: str,
    records: Sequence[Mapping[str, Any]],
    run_directory: Path,
    collator: PredictionAssistantCollator,
) -> Mapping[str, Any]:
    task_seed = _stable_seed(config.seed, "ttt-adapter", task_id)
    _seed_everything(task_seed)
    model.add_adapter(task_id, lora_config)
    adapter_added = True
    try:
        model.set_adapter(task_id)
        model.train()
        ordered_records = sorted(
            records, key=lambda record: int(record["variant_index"])
        )
        random.Random(_stable_seed(config.seed, "ttt-order", task_id)).shuffle(
            ordered_records
        )
        loader = DataLoader(
            TTTRecordDataset(ordered_records),
            batch_size=config.optimization.micro_batch_size,
            shuffle=False,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory and torch.cuda.is_available(),
            persistent_workers=config.loader.persistent_workers,
            collate_fn=collator,
        )
        if len(loader) != config.data.variants_per_task:
            raise RuntimeError(
                f"TTT task {task_id!r} resolved to {len(loader)} updates; expected "
                f"{config.data.variants_per_task}."
            )

        optimizer = build_ttt_optimizer(model, config)
        scheduler = build_ttt_scheduler(
            optimizer,
            total_steps=config.data.variants_per_task,
            warmup_ratio=config.optimization.warmup_ratio,
            warmup_start_learning_rate=(config.optimization.warmup_start_learning_rate),
            peak_learning_rate=config.profile.peak_learning_rate,
            min_learning_rate=config.optimization.min_learning_rate,
        )
        device = _model_device(model)
        dtype = _torch_dtype(config.model.dtype)
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError(f"TTT adapter {task_id!r} has no trainable parameters.")

        step_records: List[Dict[str, Any]] = []
        for step_index, (batch, source_record) in enumerate(
            zip(
                tqdm(loader, desc=f"TTT {task_id}", leave=False, unit="step"),
                ordered_records,
            ),
            start=1,
        ):
            batch = {
                name: tensor.to(device, non_blocking=config.loader.pin_memory)
                for name, tensor in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            with _autocast(device, dtype):
                output = model(**batch)
                loss = output.loss
            if loss is None or not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"TTT task {task_id!r} produced a non-finite loss at step "
                    f"{step_index}."
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                config.optimization.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            step_records.append(
                {
                    "step": step_index,
                    "variant_id": str(source_record["variant_id"]),
                    "loss": float(loss.detach().item()),
                    "gradient_norm": float(grad_norm),
                    "learning_rate": learning_rate,
                    "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "sequence_tokens": int(source_record["sequence_length"]),
                    "target_tokens": int(source_record["target_token_count"]),
                    "guidance_status": str(source_record["guidance_status"]),
                }
            )

        if len(step_records) != config.data.variants_per_task:
            raise RuntimeError(f"TTT task {task_id!r} did not complete 64 updates.")
        manifest = _task_manifest(
            task_id=task_id,
            config=config,
            task_seed=task_seed,
            records=ordered_records,
            steps=step_records,
        )
        _save_task_adapter(
            model=model,
            run_directory=run_directory,
            task_id=task_id,
            manifest=manifest,
        )
        return manifest
    finally:
        if adapter_added:
            model.set_adapter(TTT_BOOTSTRAP_ADAPTER)
            model.delete_adapter(task_id)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_ttt_adapters(
    *,
    config: TTTTrainingConfig,
    tasks_directory: Path,
    task_ids: Sequence[str],
    prediction_model_path: Path,
    output_directory: Path,
    run_name: str,
    guidance_path: Optional[Path] = None,
    augmentation_plan_path: Optional[Path] = None,
    resume: bool = False,
) -> Mapping[str, Any]:
    """Create one independently trained 64-step LoRA adapter per ARC task."""
    if not run_name.strip():
        raise ValueError("run_name must not be empty.")
    if not prediction_model_path.is_dir():
        raise FileNotFoundError(
            f"merged Prediction Model directory does not exist: "
            f"{prediction_model_path}."
        )
    if not tasks_directory.is_dir():
        raise FileNotFoundError(
            f"ARC task directory does not exist: {tasks_directory}."
        )
    normalized_task_ids = sorted(normalize_ttt_task_ids(task_ids))

    run_directory = output_directory / run_name
    if run_directory.exists() and not resume:
        raise FileExistsError(f"TTT run already exists: {run_directory}.")
    if run_directory.exists() and not run_directory.is_dir():
        raise ValueError(f"TTT run path is not a directory: {run_directory}.")

    stored_plan_path = run_directory / "augmentation_plan.json"
    plan_source = augmentation_plan_path
    if plan_source is None and resume and stored_plan_path.is_file():
        plan_source = stored_plan_path
    if plan_source is None:
        augmentation_plan = generate_ttt_augmentation_plan(
            task_ids=normalized_task_ids,
            tasks_directory=tasks_directory,
            seed=config.seed,
            variants_per_transform=config.data.variants_per_transform,
        )
    else:
        augmentation_plan = load_ttt_augmentation_plan(
            plan_source,
            task_ids=normalized_task_ids,
            tasks_directory=tasks_directory,
            variants_per_transform=config.data.variants_per_transform,
        )
    plan_payload = ttt_augmentation_plan_payload(
        augmentation_plan,
        seed=config.seed,
        variants_per_transform=config.data.variants_per_transform,
    )

    if guidance_path is not None:
        guidance_by_key = load_ttt_guidance(guidance_path)
    else:
        guidance_by_key = {}
    if (
        config.profile.guidance_mode == "guided"
        and config.profile.guidance_budget > 0
        and guidance_path is None
        and config.data.missing_guidance_policy == "error"
    ):
        raise ValueError("guided TTT with a non-zero budget requires --guidance.")

    resolved_config = json.loads(json.dumps(config.to_dict()))
    if not isinstance(resolved_config, dict):
        raise RuntimeError("TTT configuration did not serialize to a mapping.")
    resolved_config["runtime"] = {
        "prediction_model_path": str(prediction_model_path.resolve()),
        "tasks_directory": str(tasks_directory.resolve()),
        "tasks_sha256": _task_source_sha256(
            tasks_directory,
            normalized_task_ids,
        ),
        "task_ids": normalized_task_ids,
        "guidance_path": (
            str(guidance_path.resolve()) if guidance_path is not None else None
        ),
        "guidance_sha256": (
            _sha256_file(guidance_path) if guidance_path is not None else None
        ),
        "augmentation_plan_path": (
            str(augmentation_plan_path.resolve())
            if augmentation_plan_path is not None
            else None
        ),
    }
    if run_directory.exists():
        config_path = run_directory / "resolved_config.json"
        if (
            not config_path.is_file()
            or _load_json_mapping(config_path) != resolved_config
        ):
            raise ValueError("existing TTT run configuration does not match resume.")
        if not stored_plan_path.is_file():
            raise ValueError("existing TTT run has no augmentation plan.")
        if _load_json_mapping(stored_plan_path) != plan_payload:
            raise ValueError("existing TTT augmentation plan does not match resume.")

    tokenizer = load_ttt_tokenizer(prediction_model_path)
    tokenizer_metadata = _tokenizer_metadata(tokenizer)
    if run_directory.exists():
        tokenizer_path = run_directory / "tokenizer.json"
        if (
            not tokenizer_path.is_file()
            or _load_json_mapping(tokenizer_path) != tokenizer_metadata
        ):
            raise ValueError("existing TTT tokenizer metadata does not match resume.")
    records_by_task = build_ttt_records(
        config=config,
        tokenizer=tokenizer,
        tasks_directory=tasks_directory,
        task_ids=normalized_task_ids,
        augmentation_plan=augmentation_plan,
        guidance_by_key=guidance_by_key,
    )
    collator = PredictionAssistantCollator(
        pad_token_id=int(tokenizer.pad_token_id),
        ignore_first_response=config.data.ignore_first_response,
        pad_to_multiple_of=config.data.pad_to_multiple_of,
    )

    completed = (
        _completed_manifests(run_directory, normalized_task_ids)
        if run_directory.exists()
        else {}
    )
    pending_task_ids = [
        task_id for task_id in normalized_task_ids if task_id not in completed
    ]
    if not pending_task_ids:
        summary = _run_summary(
            config=config,
            task_ids=normalized_task_ids,
            manifests=completed,
        )
        _atomic_write_json(run_directory / "summary.json", summary)
        return summary

    _seed_everything(config.seed)
    model, lora_config = build_ttt_model(
        config,
        prediction_model_path=prediction_model_path,
    )
    if not run_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
        run_directory.mkdir(exist_ok=False)
        _atomic_write_json(run_directory / "resolved_config.json", resolved_config)
        _atomic_write_json(stored_plan_path, plan_payload)
        _atomic_write_json(
            run_directory / "tokenizer.json",
            tokenizer_metadata,
        )

    for task_id in pending_task_ids:
        completed[task_id] = _train_one_task(
            model=model,
            lora_config=lora_config,
            config=config,
            task_id=task_id,
            records=records_by_task[task_id],
            run_directory=run_directory,
            collator=collator,
        )
        summary = _run_summary(
            config=config,
            task_ids=normalized_task_ids,
            manifests=completed,
        )
        _atomic_write_json(run_directory / "summary.json", summary)

    return _run_summary(
        config=config,
        task_ids=normalized_task_ids,
        manifests=completed,
    )
