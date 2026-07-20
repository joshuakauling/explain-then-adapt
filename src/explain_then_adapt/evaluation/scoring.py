"""Exact-match scoring for structured Prediction Model artifacts."""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from explain_then_adapt.arc.augmented_keys import make_augmented_key
from explain_then_adapt.arc.io import load_task
from explain_then_adapt.arc.transforms import (
    inverse_transform_individual_grid,
    remap_grid_by_value_mapping,
)
from explain_then_adapt.arc.types import Grid
from explain_then_adapt.inference.artifacts import (
    manifest_path_for,
    read_jsonl,
    sha256_file,
    task_sources_sha256,
)
from explain_then_adapt.inference.config import (
    GUIDANCE_BUDGETS,
    GUIDANCE_MODES,
    INFERENCE_PROTOCOLS,
)

from .artifacts import atomic_write_json, load_verified_inference_manifest
from .config import EvaluationConfig
from .parsing import parse_prediction_grid

IDENTITY_VALUE_MAPPING = "0123456789"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _positive_integer(value: Any, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _validated_task_ids(manifest: Mapping[str, Any]) -> List[str]:
    values = manifest.get("task_ids")
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("prediction manifest task_ids must be unique strings.")
    if manifest.get("task_count") != len(values):
        raise ValueError("prediction manifest task_count does not match task_ids.")
    return values


def _validated_variant(
    record: Mapping[str, Any],
    *,
    task_id: str,
) -> Dict[str, Any]:
    variant = _mapping(record.get("variant"), "prediction record variant")
    variant_task_id = _string(variant.get("task_id"), "variant.task_id")
    if variant_task_id != task_id:
        raise ValueError("prediction record and variant task IDs disagree.")
    key = _string(variant.get("key"), "variant.key")
    is_augmented = variant.get("is_augmented")
    if not isinstance(is_augmented, bool):
        raise ValueError("variant.is_augmented must be boolean.")
    transformation_code = _string(
        variant.get("transformation_code"),
        "variant.transformation_code",
    )
    value_mapping = _string(variant.get("value_mapping"), "variant.value_mapping")
    # Applying an identity remap validates the permutation without duplicating
    # the ARC core's mapping rules.
    remap_grid_by_value_mapping([[0]], value_mapping)
    order_mapping = variant.get("order_mapping")
    variant_index = variant.get("variant_index")

    if is_augmented:
        if not isinstance(order_mapping, str) or not order_mapping:
            raise ValueError("an augmented variant requires order_mapping.")
        index = _nonnegative_integer(variant_index, "variant.variant_index")
        expected_key = make_augmented_key(
            task_id,
            transformation_code,
            value_mapping,
            order_mapping,
        )
    else:
        if (
            transformation_code != "ID"
            or value_mapping != IDENTITY_VALUE_MAPPING
            or order_mapping is not None
            or variant_index is not None
        ):
            raise ValueError("an original variant must use the identity metadata.")
        index = None
        expected_key = task_id
    if key != expected_key:
        raise ValueError(f"variant key {key!r} does not match its metadata.")
    return {
        "key": key,
        "is_augmented": is_augmented,
        "transformation_code": transformation_code,
        "value_mapping": value_mapping,
        "order_mapping": order_mapping,
        "variant_index": index,
    }


def _load_expected_outputs(
    *,
    task_ids: List[str],
    tasks_directory: Path,
) -> Dict[str, List[Grid]]:
    expected: Dict[str, List[Grid]] = {}
    for task_id in task_ids:
        task = load_task(task_id, tasks_directory)
        if not task["test"]:
            raise ValueError(f"task {task_id!r} has no test inputs.")
        outputs: List[Grid] = []
        for test_index, pair in enumerate(task["test"]):
            pair_mapping: Mapping[str, Any] = pair
            if "output" not in pair_mapping:
                raise ValueError(
                    f"task {task_id!r} test index {test_index} has no labelled output."
                )
            output = pair_mapping["output"]
            outputs.append(
                remap_grid_by_value_mapping(output, IDENTITY_VALUE_MAPPING)
            )
        expected[task_id] = outputs
    return expected


def _metric(correct: int, total: int) -> Dict[str, Any]:
    return {
        "correct": correct,
        "total": total,
        "rate": correct / total if total else 0.0,
    }


def _write_jsonl_record(file: Any, record: Mapping[str, Any]) -> None:
    json.dump(record, file, ensure_ascii=False, sort_keys=True)
    file.write("\n")


def _render_markdown_report(summary: Mapping[str, Any]) -> str:
    metrics = _mapping(summary.get("metrics"), "evaluation summary metrics")
    counts = _mapping(summary.get("counts"), "evaluation summary counts")
    labels = {
        "thesis_solve": "Thesis Solve",
        "all_test_inputs_solve": "All-Test-Inputs Solve",
        "test_input_solve": "Test-Input Solve",
        "request_solve": "Request Solve",
        "sample_accuracy": "Sample Accuracy",
        "parse_success": "Parse Success",
    }
    lines = [
        "# Evaluation Summary",
        "",
        f"- Protocol: `{summary.get('protocol')}`",
        f"- Guidance: `{summary.get('guidance_mode')}`",
        f"- Guidance budget: `{summary.get('guidance_budget')}`",
        f"- TTT enabled: `{str(summary.get('ttt_enabled')).lower()}`",
        f"- Tasks: {counts.get('tasks')}",
        f"- Test inputs: {counts.get('test_inputs')}",
        f"- Candidates: {counts.get('candidates')}",
        "",
        "## Metrics",
        "",
        "| Metric | Correct | Total | Rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in labels.items():
        metric = _mapping(metrics.get(key), f"evaluation metric {key}")
        rate = float(metric["rate"])
        lines.append(
            f"| {label} | {metric['correct']} | {metric['total']} | "
            f"{rate * 100:.2f}% |"
        )
    lines.extend(
        (
            "",
            "## Semantics",
            "",
            "`Thesis Solve` reproduces the historical Orig-Key aggregation: at "
            "least one candidate for at least one test input is correct. "
            "`All-Test-Inputs Solve` requires coverage of every test input.",
            "",
            "No candidate selector or reranker is applied. Malformed grids count "
            "as incorrect samples.",
            "",
        )
    )
    return "\n".join(lines)


def evaluate_prediction_artifact(
    *,
    config: EvaluationConfig,
    prediction_path: Path,
    tasks_directory: Path,
    output_directory: Path,
) -> Dict[str, Any]:
    """Score one complete PM run and publish auditable result artifacts."""
    manifest = load_verified_inference_manifest(
        prediction_path,
        expected_kind="prediction_inference_run",
    )
    if manifest.get("outputs_are_in_variant_space") is not True:
        raise ValueError("prediction manifest must declare variant-space outputs.")
    protocol = manifest.get("protocol")
    guidance_mode = manifest.get("guidance_mode")
    guidance_budget = manifest.get("guidance_budget")
    if protocol not in INFERENCE_PROTOCOLS:
        raise ValueError("prediction manifest protocol is invalid.")
    if guidance_mode not in GUIDANCE_MODES:
        raise ValueError("prediction manifest guidance_mode is invalid.")
    if protocol == "budgeted64":
        if guidance_mode != "guided" or guidance_budget not in GUIDANCE_BUDGETS:
            raise ValueError("budgeted64 manifest guidance settings are invalid.")
    elif guidance_budget is not None:
        raise ValueError(f"{protocol} manifest must not set a guidance budget.")
    task_ids = _validated_task_ids(manifest)
    if task_sources_sha256(tasks_directory, task_ids) != manifest.get(
        "task_sources_sha256"
    ):
        raise ValueError("current ARC task sources do not match prediction manifest.")
    expected_outputs = _load_expected_outputs(
        task_ids=task_ids,
        tasks_directory=tasks_directory,
    )

    partial_directory = output_directory.with_name(
        f".{output_directory.name}.partial"
    )
    for path in (output_directory, partial_directory):
        if path.exists():
            raise FileExistsError(f"evaluation directory already exists: {path}.")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    partial_directory.mkdir()
    candidate_path = partial_directory / "candidates.jsonl"
    task_path = partial_directory / "tasks.jsonl"
    report_path = partial_directory / "report.md"

    task_stats: Dict[str, Dict[str, Any]] = {}
    for task_id, task_outputs in expected_outputs.items():
        task_stats[task_id] = {
            "sample_count": 0,
            "parsed_sample_count": 0,
            "correct_sample_count": 0,
            "tests": {
                index: {
                    "expected_output": output,
                    "request_count": 0,
                    "sample_count": 0,
                    "parsed_sample_count": 0,
                    "correct_sample_count": 0,
                }
                for index, output in enumerate(task_outputs)
            },
        }

    seen_request_ids: Set[str] = set()
    seen_task_tests: Set[Tuple[str, int]] = set()
    variant_metadata_by_key: Dict[str, Dict[str, Any]] = {}
    variant_key_by_index: Dict[Tuple[str, int], str] = {}
    request_solved_count = 0
    candidate_count = 0
    parsed_count = 0
    correct_count = 0
    parse_statuses: Counter[str] = Counter()
    record_count = 0
    expected_samples_per_prompt = _positive_integer(
        manifest.get("samples_per_prompt"),
        "manifest.samples_per_prompt",
    )
    ttt_enabled = manifest.get("ttt_enabled")
    if not isinstance(ttt_enabled, bool):
        raise ValueError("prediction manifest ttt_enabled must be boolean.")

    with candidate_path.open("x", encoding="utf-8") as candidate_file:
        for record in read_jsonl(prediction_path):
            record_count += 1
            if record.get("schema_version") != 1:
                raise ValueError("unsupported prediction record schema version.")
            if record.get("kind") != "prediction_candidates":
                raise ValueError("prediction JSONL contains a non-candidate record.")
            if record.get("protocol") != manifest.get("protocol"):
                raise ValueError("prediction record protocol disagrees with manifest.")
            if record.get("guidance_mode") != manifest.get("guidance_mode"):
                raise ValueError(
                    "prediction record guidance mode disagrees with manifest."
                )

            request_id = _string(record.get("request_id"), "record.request_id")
            if request_id in seen_request_ids:
                raise ValueError(f"duplicate prediction request ID: {request_id!r}.")
            seen_request_ids.add(request_id)
            task_id = _string(record.get("task_id"), "record.task_id")
            if task_id not in expected_outputs:
                raise ValueError(f"prediction contains unknown task {task_id!r}.")
            test_index = _nonnegative_integer(
                record.get("test_index"),
                "record.test_index",
            )
            if test_index >= len(expected_outputs[task_id]):
                raise ValueError(
                    f"prediction test index is out of range: {request_id!r}."
                )
            variant = _validated_variant(record, task_id=task_id)
            if request_id != f"{variant['key']}__{test_index}":
                raise ValueError(
                    f"prediction request ID does not match its variant: {request_id!r}."
                )
            existing_variant = variant_metadata_by_key.get(variant["key"])
            if existing_variant is not None and existing_variant != variant:
                raise ValueError(
                    f"variant metadata changes across requests: {variant['key']!r}."
                )
            variant_metadata_by_key[variant["key"]] = variant
            if variant["variant_index"] is not None:
                index_key = (task_id, int(variant["variant_index"]))
                existing_key = variant_key_by_index.get(index_key)
                if existing_key is not None and existing_key != variant["key"]:
                    raise ValueError(
                        f"variant index {index_key!r} maps to multiple keys."
                    )
                variant_key_by_index[index_key] = str(variant["key"])
            seen_task_tests.add((task_id, test_index))

            adapter_used = record.get("adapter_used")
            if not isinstance(adapter_used, bool) or adapter_used != ttt_enabled:
                raise ValueError("record adapter usage disagrees with manifest.")
            guidance_key = record.get("guidance_key")
            if manifest.get("guidance_mode") == "guided":
                if guidance_key != variant["key"]:
                    raise ValueError("guided record has the wrong guidance key.")
            elif guidance_key is not None:
                raise ValueError("unguided record must not contain a guidance key.")

            candidate_outputs = record.get("outputs")
            if not isinstance(candidate_outputs, list):
                raise ValueError(f"prediction outputs must be a list: {request_id!r}.")
            sample_count = _positive_integer(
                record.get("sample_count"),
                "record.sample_count",
            )
            if sample_count != expected_samples_per_prompt:
                raise ValueError(
                    f"sample count disagrees with manifest: {request_id!r}."
                )
            if len(candidate_outputs) != sample_count:
                raise ValueError(
                    f"prediction sample count does not match outputs: {request_id!r}."
                )
            sample_indices = [
                _nonnegative_integer(
                    _mapping(output, "prediction output").get("sample_index"),
                    "output.sample_index",
                )
                for output in candidate_outputs
            ]
            if sample_indices != list(range(sample_count)):
                raise ValueError(
                    f"prediction sample indices are not contiguous: {request_id!r}."
                )

            test_stat = task_stats[task_id]["tests"][test_index]
            test_stat["request_count"] += 1
            request_correct = 0
            request_parsed = 0
            for output in candidate_outputs:
                output_mapping = _mapping(output, "prediction output")
                sample_index = int(output_mapping["sample_index"])
                text = output_mapping.get("text")
                if not isinstance(text, str):
                    raise ValueError("prediction output text must be a string.")
                parsed = parse_prediction_grid(
                    text,
                    max_height=config.grid_parser.max_height,
                    max_width=config.grid_parser.max_width,
                )
                parse_statuses[parsed.status] += 1
                original_grid: Optional[Grid] = None
                is_correct = False
                if parsed.grid is not None:
                    parsed_count += 1
                    request_parsed += 1
                    original_grid = inverse_transform_individual_grid(
                        parsed.grid,
                        variant["transformation_code"],
                        variant["value_mapping"],
                    )
                    is_correct = original_grid == expected_outputs[task_id][test_index]
                if is_correct:
                    request_correct += 1
                    correct_count += 1
                candidate_count += 1
                _write_jsonl_record(
                    candidate_file,
                    {
                        "schema_version": 1,
                        "kind": "scored_candidate",
                        "request_id": request_id,
                        "task_id": task_id,
                        "test_index": test_index,
                        "variant_key": variant["key"],
                        "variant_index": variant["variant_index"],
                        "sample_index": sample_index,
                        "parse_status": parsed.status,
                        "prediction_variant_space": parsed.grid,
                        "prediction_original_space": original_grid,
                        "is_correct": is_correct,
                    },
                )
            if request_correct:
                request_solved_count += 1
            task_stat = task_stats[task_id]
            task_stat["sample_count"] += sample_count
            task_stat["parsed_sample_count"] += request_parsed
            task_stat["correct_sample_count"] += request_correct
            test_stat["sample_count"] += sample_count
            test_stat["parsed_sample_count"] += request_parsed
            test_stat["correct_sample_count"] += request_correct

    if record_count != manifest.get("record_count"):
        raise ValueError("prediction record count does not match manifest.")
    if record_count != manifest.get("request_count"):
        raise ValueError("prediction request count does not match manifest.")
    if candidate_count != manifest.get("total_candidates"):
        raise ValueError("prediction candidate count does not match manifest.")
    expected_task_tests = {
        (task_id, test_index)
        for task_id, task_outputs in expected_outputs.items()
        for test_index in range(len(task_outputs))
    }
    if seen_task_tests != expected_task_tests:
        missing = sorted(expected_task_tests - seen_task_tests)
        raise ValueError(f"prediction artifact is missing test inputs: {missing}.")

    expected_candidates_per_test = _positive_integer(
        manifest.get("candidates_per_test_input"),
        "manifest.candidates_per_test_input",
    )
    for task_id, test_index in sorted(expected_task_tests):
        if (
            task_stats[task_id]["tests"][test_index]["sample_count"]
            != expected_candidates_per_test
        ):
            raise ValueError(
                f"candidate budget differs for {task_id} test index {test_index}."
            )

    thesis_solved_count = 0
    all_tests_solved_count = 0
    test_input_solved_count = 0
    with task_path.open("x", encoding="utf-8") as task_file:
        for task_id in task_ids:
            task_stat = task_stats[task_id]
            tests = []
            for test_index in sorted(task_stat["tests"]):
                test_stat = task_stat["tests"][test_index]
                solved = test_stat["correct_sample_count"] > 0
                test_input_solved_count += int(solved)
                tests.append(
                    {
                        "test_index": test_index,
                        **test_stat,
                        "solved": solved,
                    }
                )
            thesis_solved = any(test["solved"] for test in tests)
            all_tests_solved = all(test["solved"] for test in tests)
            thesis_solved_count += int(thesis_solved)
            all_tests_solved_count += int(all_tests_solved)
            _write_jsonl_record(
                task_file,
                {
                    "schema_version": 1,
                    "kind": "task_evaluation",
                    "task_id": task_id,
                    "test_input_count": len(tests),
                    "sample_count": task_stat["sample_count"],
                    "parsed_sample_count": task_stat["parsed_sample_count"],
                    "correct_sample_count": task_stat["correct_sample_count"],
                    "thesis_solved": thesis_solved,
                    "all_test_inputs_solved": all_tests_solved,
                    "test_inputs": tests,
                },
            )

    test_input_count = len(expected_task_tests)
    summary: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "evaluation_run",
        "protocol": manifest.get("protocol"),
        "guidance_mode": manifest.get("guidance_mode"),
        "guidance_budget": manifest.get("guidance_budget"),
        "ttt_enabled": manifest.get("ttt_enabled"),
        "counts": {
            "tasks": len(task_ids),
            "test_inputs": test_input_count,
            "requests": record_count,
            "candidates": candidate_count,
            "parsed_candidates": parsed_count,
            "correct_candidates": correct_count,
        },
        "metrics": {
            "thesis_solve": _metric(thesis_solved_count, len(task_ids)),
            "all_test_inputs_solve": _metric(
                all_tests_solved_count,
                len(task_ids),
            ),
            "test_input_solve": _metric(
                test_input_solved_count,
                test_input_count,
            ),
            "request_solve": _metric(request_solved_count, record_count),
            "sample_accuracy": _metric(correct_count, candidate_count),
            "parse_success": _metric(parsed_count, candidate_count),
        },
        "metric_definitions": {
            "thesis_solve": (
                "Original task has at least one correct candidate for at least "
                "one test input; this reproduces the thesis/legacy Orig-Key metric."
            ),
            "all_test_inputs_solve": (
                "Every test input of the original task has at least one correct "
                "candidate."
            ),
            "sample_accuracy": (
                "Correct output grids divided by all generated output grids."
            ),
        },
        "parse_status_counts": dict(sorted(parse_statuses.items())),
        "input": {
            "prediction_path": str(prediction_path),
            "prediction_sha256": manifest["output_sha256"],
            "prediction_manifest_path": str(manifest_path_for(prediction_path)),
            "prediction_manifest_sha256": sha256_file(
                manifest_path_for(prediction_path)
            ),
            "task_sources_sha256": manifest["task_sources_sha256"],
        },
        "config": config.to_dict(),
        "artifacts": {
            "candidates": {
                "path": "candidates.jsonl",
                "sha256": sha256_file(candidate_path),
            },
            "tasks": {
                "path": "tasks.jsonl",
                "sha256": sha256_file(task_path),
            },
        },
    }
    with report_path.open("x", encoding="utf-8") as report_file:
        report_file.write(_render_markdown_report(summary))
    summary["artifacts"]["report"] = {
        "path": "report.md",
        "sha256": sha256_file(report_path),
    }
    atomic_write_json(partial_directory / "summary.json", summary)
    partial_directory.replace(output_directory)
    return summary
