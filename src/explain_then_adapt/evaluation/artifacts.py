"""Validation and atomic JSON helpers for evaluation artifacts."""

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from explain_then_adapt.inference.artifacts import (
    manifest_path_for,
    sha256_file,
)


def load_json_mapping(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON file must contain an object: {path}.")
    return dict(value)


def load_verified_inference_manifest(
    output_path: Path,
    *,
    expected_kind: str,
) -> Dict[str, Any]:
    """Load an inference manifest and verify its published JSONL payload."""
    if not output_path.is_file():
        raise FileNotFoundError(f"inference artifact does not exist: {output_path}.")
    manifest_path = manifest_path_for(output_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"inference manifest does not exist: {manifest_path}."
        )
    manifest = load_json_mapping(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported inference manifest schema: {manifest_path}.")
    if manifest.get("kind") != expected_kind:
        raise ValueError(
            f"expected manifest kind {expected_kind!r}, got "
            f"{manifest.get('kind')!r}: {manifest_path}."
        )
    actual_sha256 = sha256_file(output_path)
    if manifest.get("output_sha256") != actual_sha256:
        raise ValueError(
            f"inference artifact hash does not match its manifest: {output_path}."
        )
    return manifest


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON object without replacing an existing result."""
    if path.exists():
        raise FileExistsError(f"evaluation output already exists: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary evaluation output exists: {temporary}.")
    with temporary.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, sort_keys=True)
        file.write("\n")
    temporary.replace(path)
