"""Deterministic migration of small reasoning-generation resources."""

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .hints import Hint, HintStatus, load_task_hint
from .records import HintMode, write_jsonl
from .validation import validate_trace_format


SCHEMA_VERSION = 1
EXPECTED_TASK_COUNT = 624
EXPECTED_LEGACY_TASK_COUNT = 391
EXPECTED_HINT_COUNT = 481
EXPECTED_TRACE_REPAIRS = {
    "insert_missing_think_close": 22,
    "remove_duplicate_summary_inside_think": 2,
}
FEW_SHOT_TASK_IDS: Tuple[str, ...] = (
    "6430c8c4",
    "7c008303",
    "08ed6ac7",
    "60b61512",
    "f25fbde4",
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_task_ids(path: Path) -> List[str]:
    value = _load_json(path)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path} must contain a list of task IDs or task paths.")
    task_ids = [Path(item).stem for item in value]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{path} contains duplicate task IDs.")
    return task_ids


def _load_traces(path: Path) -> Dict[str, str]:
    value = _load_json(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a task-ID-to-trace mapping.")
    traces: Dict[str, str] = {}
    for task_id, trace in value.items():
        if (
            not isinstance(task_id, str)
            or not isinstance(trace, str)
            or not trace.strip()
        ):
            raise ValueError(f"{path} contains an invalid trace record.")
        traces[task_id] = trace.strip()
    return traces


def _repair_legacy_trace_format(trace: str) -> Tuple[str, Optional[str]]:
    """Apply the two narrowly identified format repairs in ``624_best.json``."""
    if validate_trace_format(trace).accepted:
        return trace, None

    description_heading = "General natural language description:"
    think_close = "</think>"

    if (
        trace.count("<think>") == 1
        and trace.count(think_close) == 0
        and trace.count(description_heading) == 1
    ):
        heading_start = trace.index(description_heading)
        repaired = (
            f"{trace[:heading_start].rstrip()}\n"
            f"{think_close}\n\n"
            f"{trace[heading_start:].lstrip()}"
        )
        repair = "insert_missing_think_close"
    elif (
        trace.count("<think>") == 1
        and trace.count(think_close) == 1
        and trace.count(description_heading) == 2
    ):
        heading_start = trace.index(description_heading)
        think_close_start = trace.index(think_close)
        if heading_start >= think_close_start:
            raise ValueError("duplicate trace headings are not inside <think>.")
        repaired = (
            f"{trace[:heading_start].rstrip()}\n"
            f"{trace[think_close_start:]}"
        )
        repair = "remove_duplicate_summary_inside_think"
    else:
        raise ValueError("trace has an unsupported legacy format defect.")

    if not validate_trace_format(repaired).accepted:
        raise ValueError("legacy trace repair did not produce a valid trace.")
    return repaired, repair


def _repair_traces(
    traces: Mapping[str, str],
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    repaired_traces: Dict[str, str] = {}
    repair_records: List[Dict[str, Any]] = []
    for task_id in sorted(traces):
        trace = traces[task_id]
        try:
            repaired_trace, repair = _repair_legacy_trace_format(trace)
        except ValueError as error:
            raise ValueError(f"cannot repair trace {task_id!r}: {error}") from error
        repaired_traces[task_id] = repaired_trace
        if repair is not None:
            repair_records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "task_id": task_id,
                    "repair": repair,
                }
            )

    repair_counts = Counter(record["repair"] for record in repair_records)
    if repair_counts != Counter(EXPECTED_TRACE_REPAIRS):
        raise ValueError(
            "unexpected accepted-trace repair counts: "
            f"expected={EXPECTED_TRACE_REPAIRS}, found={dict(repair_counts)}."
        )
    return repaired_traces, repair_records


def _collect_hints(
    task_ids: Sequence[str],
    hints_directory: Path,
) -> Tuple[Dict[str, Hint], Dict[str, HintStatus]]:
    hints: Dict[str, Hint] = {}
    statuses: Dict[str, HintStatus] = {}
    for task_id in task_ids:
        result = load_task_hint(task_id, hints_directory)
        statuses[task_id] = result.status
        if result.status is HintStatus.COMPLETE:
            if result.hint is None:
                raise ValueError(f"complete hint {task_id!r} has no content.")
            hints[task_id] = result.hint
    return hints, statuses


def migrate_reasoning_resources(
    *,
    task_ids_path: Path,
    legacy_task_ids_path: Path,
    hints_directory: Path,
    traces_path: Path,
    output_directory: Path,
) -> Mapping[str, int]:
    """Build the canonical small resources from explicitly supplied legacy inputs."""
    task_ids = _load_task_ids(task_ids_path)
    legacy_task_ids = _load_task_ids(legacy_task_ids_path)
    task_id_set = set(task_ids)
    legacy_task_id_set = set(legacy_task_ids)

    if len(task_ids) != EXPECTED_TASK_COUNT:
        raise ValueError(
            f"expected {EXPECTED_TASK_COUNT} final tasks, found {len(task_ids)}."
        )
    if len(legacy_task_ids) != EXPECTED_LEGACY_TASK_COUNT:
        raise ValueError(
            "expected "
            f"{EXPECTED_LEGACY_TASK_COUNT} legacy tasks, found {len(legacy_task_ids)}."
        )
    if not legacy_task_id_set <= task_id_set:
        unexpected = sorted(legacy_task_id_set - task_id_set)
        raise ValueError(f"legacy task IDs missing from final corpus: {unexpected}.")

    hints, hint_statuses = _collect_hints(task_ids, hints_directory)
    if len(hints) != EXPECTED_HINT_COUNT:
        raise ValueError(
            f"expected {EXPECTED_HINT_COUNT} complete hints, found {len(hints)}."
        )

    legacy_traces = _load_traces(traces_path)
    if set(legacy_traces) != task_id_set:
        missing = sorted(task_id_set - set(legacy_traces))
        extra = sorted(set(legacy_traces) - task_id_set)
        raise ValueError(
            f"accepted trace IDs do not match the final corpus; missing={missing}, "
            f"extra={extra}."
        )
    traces, trace_repair_records = _repair_traces(legacy_traces)
    if not set(FEW_SHOT_TASK_IDS) <= set(hints):
        missing = sorted(set(FEW_SHOT_TASK_IDS) - set(hints))
        raise ValueError(f"few-shot tasks require complete hints: {missing}.")

    hint_records = [
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            **hints[task_id].to_dict(),
        }
        for task_id in sorted(hints)
    ]
    few_shot_records = [
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "trace": traces[task_id],
        }
        for task_id in FEW_SHOT_TASK_IDS
    ]
    base_trace_records = [
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "trace": traces[task_id],
        }
        for task_id in sorted(task_ids)
    ]
    few_shot_set: Set[str] = set(FEW_SHOT_TASK_IDS)
    manifest_records = []
    for task_id in sorted(task_ids):
        hint_status = hint_statuses[task_id]
        has_hint = hint_status is HintStatus.COMPLETE
        manifest_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "task_id": task_id,
                "corpus_partition": (
                    "legacy_training_2024"
                    if task_id in legacy_task_id_set
                    else "training_2025_addition"
                ),
                "hint_status": hint_status.value,
                "hint_mode": (
                    HintMode.PROVIDED.value if has_hint else HintMode.NONE.value
                ),
                "hint_resource_id": task_id if has_hint else None,
                "few_shot_pool": task_id in few_shot_set,
                "accepted_trace_in_legacy_source": True,
                "historical_validation_route": (
                    "unknown" if has_hint else "manual_review"
                ),
            }
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    counts = {
        "base_reasoning_traces": write_jsonl(
            base_trace_records,
            output_directory / "base_reasoning_traces.jsonl",
        ),
        "base_trace_repairs": write_jsonl(
            trace_repair_records,
            output_directory / "base_trace_repairs.jsonl",
        ),
        "hints": write_jsonl(hint_records, output_directory / "hints.jsonl"),
        "few_shot_traces": write_jsonl(
            few_shot_records,
            output_directory / "few_shot_traces.jsonl",
        ),
        "tasks": write_jsonl(
            manifest_records,
            output_directory / "task_manifest.jsonl",
        ),
    }
    return counts
