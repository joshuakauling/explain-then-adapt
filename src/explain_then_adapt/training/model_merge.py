"""Merge a trained LoRA adapter into a non-quantized base model."""

import json
from pathlib import Path
from typing import Any, Dict

import torch


def torch_dtype(name: str) -> torch.dtype:
    try:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[name]
    except KeyError as error:
        raise ValueError("dtype must be bfloat16, float16, or float32.") from error


def merge_lora_adapter(
    *,
    base_model: str,
    adapter_path: Path,
    output_directory: Path,
    dtype: str,
    device_map: str,
    max_shard_size: str,
    trust_remote_code: bool,
) -> Dict[str, Any]:
    """Merge one PEFT adapter and save a standalone Transformers model."""
    if not base_model.strip():
        raise ValueError("base_model must not be empty.")
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"adapter directory does not exist: {adapter_path}.")
    if output_directory.exists():
        raise FileExistsError(f"merge output already exists: {output_directory}.")
    if not device_map.strip() or not max_shard_size.strip():
        raise ValueError("device_map and max_shard_size must not be empty.")
    try:
        from peft import PeftModel  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "install the optional training dependencies with "
            "`python -m pip install -e '.[training]'`."
        ) from error

    resolved_dtype = torch_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=resolved_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    merged = model.merge_and_unload()
    merged.eval()

    output_directory.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(output_directory)
    merged.save_pretrained(
        output_directory,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    manifest: Dict[str, Any] = {
        "base_model": base_model,
        "adapter_path": str(adapter_path),
        "dtype": dtype,
        "device_map": device_map,
        "max_shard_size": max_shard_size,
        "trust_remote_code": trust_remote_code,
    }
    with (output_directory / "merge_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    return manifest
