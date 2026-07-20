"""Migration of the final Reasoning Model validation resources."""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Set

from explain_then_adapt.arc.augmented_keys import parse_augmented_key
from explain_then_adapt.data_generation.records import read_jsonl, write_jsonl
from explain_then_adapt.data_generation.validation import validate_trace_format

SCHEMA_VERSION = 1
EXPECTED_VALIDATION_TASKS = 39


def _training_task_ids(path: Path) -> Set[str]:
    task_ids = {str(record.get("task_id", "")) for record in read_jsonl(path)}
    if "" in task_ids:
        raise ValueError("training task manifest contains an empty task_id.")
    return task_ids


def _trace(record: Mapping[str, Any], source: Path) -> str:
    value = record.get("text")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source} contains a record without trace text.")
    trace = value.strip()
    if not validate_trace_format(trace).accepted:
        key = record.get("key")
        raise ValueError(f"validation trace {key!r} in {source} is malformed.")
    return trace


def migrate_reasoning_validation_resources(
    *,
    validation_path: Path,
    augmented_validation_path: Path,
    training_task_manifest_path: Path,
    output_directory: Path,
) -> Mapping[str, int]:
    """Build the two final, disjoint 39-task validation resources."""
    training_ids = _training_task_ids(training_task_manifest_path)
    original_records: List[Dict[str, Any]] = []
    original_ids: Set[str] = set()
    for source_record in read_jsonl(validation_path):
        key = source_record.get("key")
        if not isinstance(key, str) or not key or "_" in key:
            raise ValueError("original validation keys must be plain ARC task IDs.")
        if key in original_ids:
            raise ValueError(f"duplicate original validation task: {key!r}.")
        original_ids.add(key)
        original_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "split": "validation",
                "variant_id": key,
                "task_id": key,
                "trace": _trace(source_record, validation_path),
                "historical_validation_route": "manual_review",
            }
        )

    augmented_records: List[Dict[str, Any]] = []
    augmented_ids: Set[str] = set()
    seen_variant_ids: Set[str] = set()
    for source_record in read_jsonl(augmented_validation_path):
        key = source_record.get("key")
        if not isinstance(key, str):
            raise ValueError("augmented validation records require a string key.")
        parsed = parse_augmented_key(key)
        if (
            parsed.transformation_id is None
            or parsed.value_mapping is None
            or parsed.order_mapping is None
        ):
            raise ValueError(f"augmented validation key is not augmented: {key!r}.")
        if key in seen_variant_ids:
            raise ValueError(f"duplicate augmented validation variant: {key!r}.")
        if parsed.original_key in augmented_ids:
            raise ValueError(
                f"duplicate augmented validation task: {parsed.original_key!r}."
            )
        seen_variant_ids.add(key)
        augmented_ids.add(parsed.original_key)
        augmented_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "split": "validation_augmented",
                "variant_id": key,
                "task_id": parsed.original_key,
                "augmentation": {
                    "transformation_code": parsed.transformation_id,
                    "value_mapping": parsed.value_mapping,
                    "order_mapping": parsed.order_mapping,
                },
                "trace": _trace(source_record, augmented_validation_path),
                "historical_validation_route": "manual_review",
            }
        )

    if len(original_ids) != EXPECTED_VALIDATION_TASKS:
        raise ValueError(
            f"expected {EXPECTED_VALIDATION_TASKS} original validation tasks, "
            f"found {len(original_ids)}."
        )
    if augmented_ids != original_ids:
        raise ValueError(
            "original and augmented validation resources must cover the same tasks."
        )
    overlap = sorted(original_ids & training_ids)
    if overlap:
        raise ValueError(f"validation tasks overlap the training manifest: {overlap}.")

    output_directory.mkdir(parents=True, exist_ok=True)
    return {
        "validation": write_jsonl(
            sorted(original_records, key=lambda record: record["task_id"]),
            output_directory / "reasoning_validation.jsonl",
        ),
        "validation_augmented": write_jsonl(
            sorted(augmented_records, key=lambda record: record["task_id"]),
            output_directory / "reasoning_validation_augmented.jsonl",
        ),
    }
