"""Token accounting for the thesis seconds-equivalent compute measure."""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from explain_then_adapt.inference.artifacts import manifest_path_for, sha256_file

from .artifacts import load_json_mapping, load_verified_inference_manifest
from .config import EvaluationConfig


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _task_ids(manifest: Mapping[str, Any], name: str) -> List[str]:
    values = manifest.get("task_ids")
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name}.task_ids must be a non-empty unique string list.")
    if manifest.get("task_count") != len(values):
        raise ValueError(f"{name}.task_count does not match task_ids.")
    return values


def _optional_artifact_sha(value: Any, name: str) -> Optional[str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an artifact metadata object.")
    path = value.get("path")
    digest = value.get("sha256")
    if path is None and digest is None:
        return None
    if not isinstance(path, str) or not path:
        raise ValueError(f"{name}.path must be a non-empty string or null.")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{name}.sha256 must be a SHA-256 digest or null.")
    return digest


def _load_ttt_run(
    run_directory: Path,
    *,
    expected_task_ids: Sequence[str],
    expected_task_sources_sha256: str,
) -> Dict[str, Any]:
    summary = load_json_mapping(run_directory / "summary.json")
    resolved = load_json_mapping(run_directory / "resolved_config.json")
    if summary.get("schema_version") != 1 or summary.get("kind") != "ttt_run":
        raise ValueError(f"invalid TTT summary: {run_directory / 'summary.json'}.")
    completed = summary.get("completed_tasks")
    if (
        not isinstance(completed, list)
        or any(not isinstance(task_id, str) for task_id in completed)
        or sorted(completed) != sorted(expected_task_ids)
    ):
        raise ValueError("TTT summary does not cover exactly the prediction tasks.")
    if summary.get("completed_task_count") != len(expected_task_ids):
        raise ValueError("TTT summary is incomplete.")
    if summary.get("task_count") != len(expected_task_ids):
        raise ValueError("TTT summary task_count does not match the prediction run.")
    runtime = resolved.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("TTT resolved_config.runtime must be an object.")
    if runtime.get("tasks_sha256") != expected_task_sources_sha256:
        raise ValueError("TTT and prediction task sources do not match.")
    guidance_sha256 = runtime.get("guidance_sha256")
    if guidance_sha256 is not None and (
        not isinstance(guidance_sha256, str) or len(guidance_sha256) != 64
    ):
        raise ValueError("TTT runtime guidance_sha256 is invalid.")
    guidance_mode = summary.get("guidance_mode")
    guidance_budget = summary.get("guidance_budget")
    if guidance_mode not in {"guided", "unguided"}:
        raise ValueError("TTT summary guidance_mode is invalid.")
    guidance_budget = _nonnegative_integer(
        guidance_budget,
        "TTT summary guidance_budget",
    )
    uses_reasoning = guidance_mode == "guided" and guidance_budget > 0
    if uses_reasoning and guidance_sha256 is None:
        raise ValueError("guided TTT has no provenance-linked reasoning artifact.")
    return {
        "processed_tokens": _nonnegative_integer(
            summary.get("processed_tokens"),
            "TTT summary processed_tokens",
        ),
        "guidance_sha256": guidance_sha256 if uses_reasoning else None,
        "summary_sha256": sha256_file(run_directory / "summary.json"),
        "resolved_config_sha256": sha256_file(
            run_directory / "resolved_config.json"
        ),
    }


def summarize_compute(
    *,
    config: EvaluationConfig,
    prediction_path: Path,
    reasoning_paths: Sequence[Path] = (),
    ttt_run_directory: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compute a complete, provenance-checked token budget for one PM run."""
    prediction_manifest = load_verified_inference_manifest(
        prediction_path,
        expected_kind="prediction_inference_run",
    )
    prediction_task_ids = _task_ids(prediction_manifest, "prediction manifest")
    task_sources_sha256 = prediction_manifest.get("task_sources_sha256")
    if not isinstance(task_sources_sha256, str) or len(task_sources_sha256) != 64:
        raise ValueError("prediction manifest task_sources_sha256 is invalid.")

    required_reasoning: Set[str] = set()
    prediction_guidance_sha = _optional_artifact_sha(
        prediction_manifest.get("guidance_artifact"),
        "prediction manifest guidance_artifact",
    )
    guidance_mode = prediction_manifest.get("guidance_mode")
    if guidance_mode not in {"guided", "unguided"}:
        raise ValueError("prediction manifest guidance_mode is invalid.")
    if guidance_mode == "guided" and prediction_guidance_sha is None:
        raise ValueError("guided prediction has no reasoning-artifact provenance.")
    if guidance_mode == "unguided" and prediction_guidance_sha is not None:
        raise ValueError("unguided prediction must not reference reasoning guidance.")
    if prediction_guidance_sha is not None:
        required_reasoning.add(prediction_guidance_sha)

    ttt_enabled = prediction_manifest.get("ttt_enabled")
    if not isinstance(ttt_enabled, bool):
        raise ValueError("prediction manifest ttt_enabled must be boolean.")
    ttt_tokens = 0
    ttt_source: Optional[Dict[str, Any]] = None
    if ttt_enabled:
        if ttt_run_directory is None:
            raise ValueError(
                "a TTT-enabled prediction requires --ttt-run for complete cost "
                "accounting."
            )
        ttt_source = _load_ttt_run(
            ttt_run_directory,
            expected_task_ids=prediction_task_ids,
            expected_task_sources_sha256=task_sources_sha256,
        )
        ttt_tokens = int(ttt_source["processed_tokens"])
        if ttt_source["guidance_sha256"] is not None:
            required_reasoning.add(str(ttt_source["guidance_sha256"]))
    elif ttt_run_directory is not None:
        raise ValueError("a non-TTT prediction must not receive --ttt-run.")

    reasoning_prompt_tokens = 0
    reasoning_generated_tokens = 0
    provided_reasoning: Set[str] = set()
    reasoning_sources: List[Dict[str, Any]] = []
    for reasoning_path in reasoning_paths:
        manifest = load_verified_inference_manifest(
            reasoning_path,
            expected_kind="reasoning_inference_run",
        )
        digest = manifest.get("output_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("reasoning manifest output_sha256 is invalid.")
        if digest in provided_reasoning:
            raise ValueError("the same reasoning artifact was provided more than once.")
        provided_reasoning.add(digest)
        if set(_task_ids(manifest, "reasoning manifest")) != set(prediction_task_ids):
            raise ValueError("reasoning and prediction task IDs do not match.")
        if manifest.get("task_sources_sha256") != task_sources_sha256:
            raise ValueError("reasoning and prediction task sources do not match.")
        prompt_tokens = _nonnegative_integer(
            manifest.get("total_prompt_tokens"),
            "reasoning manifest total_prompt_tokens",
        )
        generated_tokens = _nonnegative_integer(
            manifest.get("total_generated_tokens"),
            "reasoning manifest total_generated_tokens",
        )
        reasoning_prompt_tokens += prompt_tokens
        reasoning_generated_tokens += generated_tokens
        reasoning_sources.append(
            {
                "path": str(reasoning_path),
                "sha256": digest,
                "manifest_path": str(manifest_path_for(reasoning_path)),
                "manifest_sha256": sha256_file(
                    manifest_path_for(reasoning_path)
                ),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
            }
        )
    if provided_reasoning != required_reasoning:
        missing = sorted(required_reasoning - provided_reasoning)
        unexpected = sorted(provided_reasoning - required_reasoning)
        raise ValueError(
            "reasoning artifacts do not match prediction/TTT provenance; "
            f"missing={missing}, unexpected={unexpected}."
        )

    prediction_prompt_tokens = _nonnegative_integer(
        prediction_manifest.get("total_prompt_tokens"),
        "prediction manifest total_prompt_tokens",
    )
    prediction_generated_tokens = _nonnegative_integer(
        prediction_manifest.get("total_generated_tokens"),
        "prediction manifest total_generated_tokens",
    )
    prefill_tokens = reasoning_prompt_tokens + prediction_prompt_tokens
    generated_tokens = reasoning_generated_tokens + prediction_generated_tokens
    compute = config.compute
    prefill_seconds = prefill_tokens / compute.prefill_tokens_per_second
    training_seconds = (
        compute.training_token_multiplier
        * ttt_tokens
        / compute.prefill_tokens_per_second
    )
    decode_seconds = generated_tokens / compute.decode_tokens_per_second
    total_seconds = prefill_seconds + training_seconds + decode_seconds

    return {
        "schema_version": 1,
        "kind": "compute_summary",
        "protocol": prediction_manifest.get("protocol"),
        "guidance_mode": prediction_manifest.get("guidance_mode"),
        "guidance_budget": prediction_manifest.get("guidance_budget"),
        "task_count": len(prediction_task_ids),
        "tokens": {
            "reasoning_prompt": reasoning_prompt_tokens,
            "reasoning_generated": reasoning_generated_tokens,
            "ttt_train": ttt_tokens,
            "prediction_prompt": prediction_prompt_tokens,
            "prediction_generated": prediction_generated_tokens,
            "prefill_total": prefill_tokens,
            "generated_total": generated_tokens,
        },
        "seconds_equivalent": {
            "prefill": prefill_seconds,
            "training": training_seconds,
            "decode": decode_seconds,
            "total": total_seconds,
        },
        "hours_equivalent": total_seconds / 3600.0,
        "assumptions": {
            "prefill_tokens_per_second": compute.prefill_tokens_per_second,
            "decode_tokens_per_second": compute.decode_tokens_per_second,
            "training_token_multiplier": compute.training_token_multiplier,
        },
        "sources": {
            "prediction": {
                "path": str(prediction_path),
                "sha256": prediction_manifest["output_sha256"],
                "manifest_path": str(manifest_path_for(prediction_path)),
                "manifest_sha256": sha256_file(
                    manifest_path_for(prediction_path)
                ),
            },
            "reasoning": reasoning_sources,
            "ttt": (
                {
                    "run_directory": str(ttt_run_directory),
                    **ttt_source,
                }
                if ttt_source is not None
                else None
            ),
        },
    }
