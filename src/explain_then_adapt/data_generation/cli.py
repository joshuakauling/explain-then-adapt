"""Command-line orchestration for the offline reasoning-data pipeline."""

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from explain_then_adapt.arc.io import load_puzzle_train

from .augmentation import augmentation_signature, plan_augmentation_specs
from .backends.gemini import (
    download_batch_results,
    get_batch,
    read_gemini_batch_results,
    submit_batch,
    write_gemini_batch_input,
)
from .backends.vllm import run_vllm_requests
from .hints import Hint, HintStatus, load_hints_jsonl, load_task_hint
from .pipeline import (
    apply_static_validation,
    build_initial_request,
    build_judge_requests,
    build_rewrite_request,
)
from .prompts import FewShotExample
from .records import (
    AugmentationSpec,
    GenerationRequest,
    GenerationResult,
    GenerationStage,
    SamplingParameters,
    read_jsonl,
    read_requests,
    read_results,
    write_requests,
    write_results,
)
from .validation import (
    JudgeVerdict,
    evaluate_judge_responses,
    parse_judge_verdict,
    record_manual_review,
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_task_ids(path: Path) -> List[str]:
    if path.suffix == ".jsonl":
        values = [record.get("task_id") for record in read_jsonl(path)]
        if not all(isinstance(item, str) and item.strip() for item in values):
            raise ValueError("every task-manifest record requires a non-empty task_id.")
        task_ids = [str(item) for item in values]
    else:
        value = _load_json(path)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError("task-id JSON must contain a list of strings.")
        task_ids = [Path(item).stem for item in value]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task input contains duplicate task IDs.")
    return task_ids


def _load_trace(entry: Mapping[str, Any], manifest_directory: Path) -> str:
    trace = entry.get("trace")
    if isinstance(trace, str) and trace.strip():
        return trace.strip()
    trace_path = entry.get("trace_path")
    if not isinstance(trace_path, str) or not trace_path.strip():
        raise ValueError("few-shot entries require 'trace' or 'trace_path'.")
    path = Path(trace_path)
    if not path.is_absolute():
        path = manifest_directory / path
    trace_key = entry.get("trace_key", entry.get("task_id"))
    if path.suffix == ".jsonl":
        if not isinstance(trace_key, str):
            raise ValueError("JSONL trace lookup requires a task_id or trace_key.")
        matches = [
            record
            for record in read_jsonl(path)
            if record.get("task_id") == trace_key
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one trace for {trace_key!r} in {path}, found {len(matches)}."
            )
        value: Any = matches[0].get("trace")
    else:
        value = (
            _load_json(path)
            if path.suffix == ".json"
            else path.read_text(encoding="utf-8")
        )
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, Mapping):
        if isinstance(trace_key, str) and isinstance(value.get(trace_key), str):
            value = value[trace_key]
        else:
            value = next(
                (
                    value[key]
                    for key in ("trace", "cot", "response", "final")
                    if key in value
                ),
                None,
            )
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"could not load a reasoning trace from {path}.")
    return value.strip()


HintSource = Union[Path, Mapping[str, Hint]]


def _load_hint_source(path: Optional[Path]) -> Optional[HintSource]:
    if path is None:
        return None
    if path.is_dir():
        return path
    if path.is_file() and path.suffix == ".jsonl":
        return load_hints_jsonl(path)
    raise ValueError("--hints must point to a hint directory or a JSONL resource.")


def _optional_hint(
    task_id: str,
    hint_source: Optional[HintSource],
) -> Tuple[Optional[Hint], str]:
    if hint_source is None:
        return None, HintStatus.MISSING.value
    if isinstance(hint_source, Mapping):
        hint = hint_source.get(task_id)
        if hint is None:
            return None, HintStatus.MISSING.value
        return hint, HintStatus.COMPLETE.value
    loaded = load_task_hint(task_id, hint_source)
    return loaded.hint, loaded.status.value


def _load_few_shot_pool(
    manifest_path: Path,
    *,
    tasks_directory: Path,
    hint_source: Optional[HintSource],
) -> Dict[str, FewShotExample]:
    value: Any = (
        read_jsonl(manifest_path)
        if manifest_path.suffix == ".jsonl"
        else _load_json(manifest_path)
    )
    if not isinstance(value, list):
        raise ValueError("few-shot manifest must contain a list of objects.")
    examples: Dict[str, FewShotExample] = {}
    for entry in value:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("task_id"), str):
            raise ValueError("every few-shot entry requires a string 'task_id'.")
        task_id = str(entry["task_id"])
        if task_id in examples:
            raise ValueError(f"duplicate few-shot task_id: {task_id!r}.")
        hint, _ = _optional_hint(task_id, hint_source)
        examples[task_id] = FewShotExample(
            task_id=task_id,
            puzzle=load_puzzle_train(task_id, tasks_directory),
            trace=_load_trace(entry, manifest_path.parent),
            hint=hint,
        )
    return examples


def _stable_seed(seed: int, *parts: str) -> int:
    payload = ":".join((str(seed),) + parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _few_shot_seed(base_seed: int, task_id: str, candidate_index: int) -> int:
    """Reproduce the per-task seed used by the final legacy few-shot sampler."""
    task_hash = int.from_bytes(
        hashlib.sha256(task_id.encode("utf-8")).digest()[:4],
        "big",
    )
    per_task_seed = (base_seed + task_hash) & ((1 << 32) - 1)
    return per_task_seed + candidate_index


def _select_few_shots(
    pool: Mapping[str, FewShotExample],
    *,
    task_id: str,
    count: int,
    seed: int,
) -> Tuple[FewShotExample, ...]:
    candidates = [key for key in pool if key != task_id]
    if count > len(candidates):
        raise ValueError(
            f"task {task_id!r} needs {count} few-shots, but only "
            f"{len(candidates)} are available."
        )
    import random

    selected = random.Random(seed).sample(candidates, count)
    return tuple(pool[key] for key in selected)


def _sampling_from_args(args: argparse.Namespace) -> SamplingParameters:
    return SamplingParameters(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.sampling_seed,
        use_provider_defaults=getattr(args, "provider_defaults", False),
    )


def _with_hint_status(
    request: GenerationRequest,
    hint_status: str,
) -> GenerationRequest:
    metadata = dict(request.metadata)
    metadata["hint_status"] = hint_status
    return replace(request, metadata=metadata)


def prepare_initial(args: argparse.Namespace) -> None:
    if args.candidate_start < 0:
        raise ValueError("--candidate-start must be non-negative.")
    if args.candidates_per_task <= 0:
        raise ValueError("--candidates-per-task must be positive.")
    task_ids = _load_task_ids(args.task_ids)
    hint_source = _load_hint_source(args.hints)
    pool = _load_few_shot_pool(
        args.few_shot_manifest,
        tasks_directory=args.tasks_dir,
        hint_source=hint_source,
    )
    requests: List[GenerationRequest] = []
    for task_id in task_ids:
        puzzle = load_puzzle_train(task_id, args.tasks_dir)
        hint, hint_status = _optional_hint(task_id, hint_source)
        candidate_stop = args.candidate_start + args.candidates_per_task
        for candidate_index in range(args.candidate_start, candidate_stop):
            selection_seed = _few_shot_seed(args.seed, task_id, candidate_index)
            request = build_initial_request(
                task_id=task_id,
                puzzle=puzzle,
                hint=hint,
                few_shots=_select_few_shots(
                    pool,
                    task_id=task_id,
                    count=args.few_shots_per_request,
                    seed=selection_seed,
                ),
                sampling=_sampling_from_args(args),
                candidate_index=candidate_index,
            )
            requests.append(_with_hint_status(request, hint_status))
    write_requests(requests, args.output)
    print(f"wrote {len(requests)} initial requests to {args.output}")


def validate_static(args: argparse.Namespace) -> None:
    validated: List[GenerationResult] = []
    for result in read_results(args.input):
        if result.stage is GenerationStage.JUDGE:
            raise ValueError("static trace validation does not apply to judge results.")
        validated.append(apply_static_validation(result))
    write_results(validated, args.output)
    accepted = sum(
        bool(result.validation and result.validation["static"]["accepted"])
        for result in validated
    )
    print(f"wrote {len(validated)} results; {accepted} passed static validation")


def render_requests(args: argparse.Namespace) -> None:
    requests = read_requests(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for request in requests:
        sections = [
            f"request_id: {request.request_id}",
            f"task_id: {request.task_id}",
            f"stage: {request.stage.value}",
            f"prompt_version: {request.prompt_version}",
            f"hint_mode: {request.hint_mode.value}",
            "few_shot_task_ids: " + ", ".join(request.few_shot_task_ids),
        ]
        for message in request.messages:
            sections.append(f"## {message.role.upper()}\n\n{message.content.rstrip()}")
        output_path = args.output_dir / f"{request.request_id}.txt"
        output_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"rendered {len(requests)} requests to {args.output_dir}")


def prepare_judge(args: argparse.Namespace) -> None:
    hint_source = _load_hint_source(args.hints)
    requests: List[GenerationRequest] = []
    for source in read_results(args.input):
        if source.stage is not GenerationStage.INITIAL or source.error:
            continue
        validated = source if source.validation else apply_static_validation(source)
        if not _static_accepted(validated):
            continue
        hint, hint_status = _optional_hint(source.task_id, hint_source)
        trace = validated.normalized_output or validated.raw_output
        new_requests = build_judge_requests(
            task_id=source.task_id,
            puzzle=load_puzzle_train(source.task_id, args.tasks_dir),
            candidate_trace=trace,
            source_request_id=source.request_id,
            hint=hint,
            sampling=_sampling_from_args(args),
        )
        requests.extend(
            _with_hint_status(request, hint_status) for request in new_requests
        )
    write_requests(requests, args.output)
    print(f"wrote {len(requests)} judge requests to {args.output}")


def evaluate_judges(args: argparse.Namespace) -> None:
    source_results = {
        result.request_id: result for result in read_results(args.sources)
    }
    judge_requests = read_requests(args.judge_requests)
    judge_results = {
        result.request_id: result for result in read_results(args.judge_results)
    }
    grouped: Dict[str, List[GenerationRequest]] = {}
    for request in judge_requests:
        source_id = str(request.metadata.get("source_request_id", ""))
        grouped.setdefault(source_id, []).append(request)

    evaluated: List[GenerationResult] = []
    for source_id, source in source_results.items():
        requests = sorted(
            grouped.get(source_id, []),
            key=lambda request: int(request.metadata.get("judge_index", -1)),
        )
        if not requests:
            continue
        if len(requests) != 5:
            raise ValueError(
                f"source {source_id!r} has {len(requests)} judge requests."
            )

        verdicts: List[Mapping[str, str]] = []
        parse_errors: List[str] = []
        for request in requests:
            if request.request_id not in judge_results:
                raise ValueError(f"missing judge result for {request.request_id!r}.")
            result = judge_results[request.request_id]
            try:
                verdict = parse_judge_verdict(result.raw_output)
            except (TypeError, ValueError) as error:
                verdict = JudgeVerdict.FAIL
                parse_errors.append(f"{request.request_id}: {error}")
            verdicts.append({"verdict": verdict.value})

        quality = evaluate_judge_responses(verdicts)
        validation = dict(source.validation or {})
        validation["quality"] = dict(quality.to_dict())
        if parse_errors:
            validation["judge_parse_errors"] = parse_errors
        evaluated.append(replace(source, validation=validation))

    write_results(evaluated, args.output)
    accepted = sum(
        bool(result.validation and result.validation["quality"]["accepted"])
        for result in evaluated
    )
    print(f"wrote {len(evaluated)} decisions; {accepted} passed all five judges")


def apply_manual_reviews(args: argparse.Namespace) -> None:
    decisions = {
        str(value["request_id"]): value for value in read_jsonl(args.decisions)
    }
    reviewed: List[GenerationResult] = []
    for source in read_results(args.sources):
        decision = decisions.get(source.request_id)
        if decision is None:
            reviewed.append(source)
            continue
        accepted = decision.get("accepted")
        if not isinstance(accepted, bool):
            raise ValueError(
                f"manual decision for {source.request_id!r} requires boolean "
                "'accepted'."
            )
        quality = record_manual_review(
            accepted=accepted,
            reviewer_note=str(decision.get("reviewer_note", "")),
        )
        validation = dict(source.validation or {})
        validation["quality"] = dict(quality.to_dict())
        reviewed.append(replace(source, validation=validation))
    write_results(reviewed, args.output)
    print(f"wrote {len(reviewed)} source records with explicit manual decisions")


def _quality_accepted(result: GenerationResult) -> bool:
    validation = result.validation or {}
    quality = validation.get("quality", {})
    return (
        _static_accepted(result)
        and isinstance(quality, Mapping)
        and quality.get("accepted") is True
    )


def _static_accepted(result: Optional[GenerationResult]) -> bool:
    if result is None or result.error:
        return False
    validation = result.validation or {}
    static = validation.get("static", {})
    return isinstance(static, Mapping) and bool(static.get("accepted", False))


def prepare_rewrite(args: argparse.Namespace) -> None:
    sources = [
        result for result in read_results(args.sources) if _quality_accepted(result)
    ]
    existing_requests = (
        read_requests(args.existing_requests) if args.existing_requests else []
    )
    existing_results = (
        {result.request_id: result for result in read_results(args.existing_results)}
        if args.existing_results
        else {}
    )
    requests_by_source: Dict[str, List[GenerationRequest]] = {}
    for request in existing_requests:
        if request.augmentation is not None:
            requests_by_source.setdefault(
                request.augmentation.source_trace_id,
                [],
            ).append(request)

    rewrite_requests: List[GenerationRequest] = []
    for source in sources:
        prior_requests = requests_by_source.get(source.request_id, [])
        accepted_specs = [
            request.augmentation
            for request in prior_requests
            if request.augmentation is not None
            and _static_accepted(existing_results.get(request.request_id))
        ]
        accepted_signatures = {
            augmentation_signature(spec) for spec in accepted_specs
        }
        attempted_specs = [
            request.augmentation
            for request in prior_requests
            if request.augmentation is not None
            and augmentation_signature(request.augmentation) not in accepted_signatures
        ]
        specs = plan_augmentation_specs(
            source_trace_id=source.request_id,
            example_count=len(load_puzzle_train(source.task_id, args.tasks_dir)),
            accepted_specs=accepted_specs,
            attempted_specs=attempted_specs,
            target_count=args.target_count,
            seed=_stable_seed(args.seed, source.task_id, source.request_id),
            styles=args.styles,
        )
        puzzle = load_puzzle_train(source.task_id, args.tasks_dir)
        trace = source.normalized_output or source.raw_output
        rewrite_requests.extend(
            build_rewrite_request(
                task_id=source.task_id,
                puzzle=puzzle,
                accepted_trace=trace,
                spec=spec,
                sampling=_sampling_from_args(args),
            )
            for spec in specs
        )

    write_requests(rewrite_requests, args.output)
    print(f"wrote {len(rewrite_requests)} rewrite requests to {args.output}")


def gemini_export(args: argparse.Namespace) -> None:
    count = write_gemini_batch_input(read_requests(args.input), args.output)
    print(f"wrote {count} Gemini batch requests to {args.output}")


def gemini_import(args: argparse.Namespace) -> None:
    results = read_gemini_batch_results(
        args.input,
        read_requests(args.requests),
        model=args.model,
    )
    write_results(results, args.output)
    print(f"wrote {len(results)} normalized Gemini results to {args.output}")


def gemini_submit(args: argparse.Namespace) -> None:
    batch = submit_batch(
        args.input,
        model=args.model,
        display_name=args.display_name,
    )
    print(json.dumps({"name": getattr(batch, "name", None)}, indent=2))


def gemini_status(args: argparse.Namespace) -> None:
    batch = get_batch(args.batch_name)
    state = getattr(batch, "state", None)
    print(
        json.dumps(
            {
                "name": getattr(batch, "name", args.batch_name),
                "state": getattr(state, "name", str(state)),
            },
            indent=2,
        )
    )


def gemini_download(args: argparse.Namespace) -> None:
    path = download_batch_results(
        args.batch_name,
        args.output,
    )
    print(f"downloaded Gemini batch results to {path}")


def vllm_run(args: argparse.Namespace) -> None:
    requests = read_requests(args.input)
    results = run_vllm_requests(
        requests,
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
    )
    write_results(results, args.output)
    print(f"wrote {len(results)} vLLM results to {args.output}")


def _add_sampling_arguments(
    parser: argparse.ArgumentParser,
    *,
    temperature: float,
    max_tokens: int,
    allow_provider_defaults: bool = False,
) -> None:
    parser.add_argument("--temperature", type=float, default=temperature)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=max_tokens)
    parser.add_argument("--sampling-seed", type=int)
    if allow_provider_defaults:
        parser.add_argument(
            "--provider-defaults",
            action="store_true",
            help="omit generation_config and use the remote model defaults",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initial = subparsers.add_parser("prepare-initial", help="build initial requests")
    initial.add_argument("--tasks-dir", type=Path, required=True)
    initial.add_argument("--task-ids", type=Path, required=True)
    initial.add_argument(
        "--hints",
        "--hints-dir",
        dest="hints",
        type=Path,
        help="hint directory or versioned hints JSONL",
    )
    initial.add_argument("--few-shot-manifest", type=Path, required=True)
    initial.add_argument("--few-shots-per-request", type=int, default=2)
    initial.add_argument(
        "--candidate-start",
        type=int,
        default=0,
        help="first candidate index, useful when resuming retries",
    )
    initial.add_argument("--candidates-per-task", type=int, default=1)
    initial.add_argument("--seed", type=int, default=0)
    initial.add_argument("--output", type=Path, required=True)
    _add_sampling_arguments(
        initial,
        temperature=0.7,
        max_tokens=8192,
        allow_provider_defaults=True,
    )
    initial.set_defaults(handler=prepare_initial)

    static = subparsers.add_parser("validate-static", help="validate trace schemas")
    static.add_argument("--input", type=Path, required=True)
    static.add_argument("--output", type=Path, required=True)
    static.set_defaults(handler=validate_static)

    render = subparsers.add_parser(
        "render-requests",
        help="render request messages as readable text files",
    )
    render.add_argument("--input", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)
    render.set_defaults(handler=render_requests)

    judge = subparsers.add_parser("prepare-judge", help="build five judge requests")
    judge.add_argument("--input", type=Path, required=True)
    judge.add_argument("--tasks-dir", type=Path, required=True)
    judge.add_argument(
        "--hints",
        "--hints-dir",
        dest="hints",
        type=Path,
        help="hint directory or versioned hints JSONL",
    )
    judge.add_argument("--output", type=Path, required=True)
    _add_sampling_arguments(judge, temperature=0.7, max_tokens=2048)
    judge.set_defaults(handler=prepare_judge)

    evaluate = subparsers.add_parser("evaluate-judges", help="apply the 5/5 rule")
    evaluate.add_argument("--sources", type=Path, required=True)
    evaluate.add_argument("--judge-requests", type=Path, required=True)
    evaluate.add_argument("--judge-results", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(handler=evaluate_judges)

    manual = subparsers.add_parser(
        "apply-manual-reviews",
        help="attach explicit manual-review decisions",
    )
    manual.add_argument("--sources", type=Path, required=True)
    manual.add_argument("--decisions", type=Path, required=True)
    manual.add_argument("--output", type=Path, required=True)
    manual.set_defaults(handler=apply_manual_reviews)

    rewrite = subparsers.add_parser(
        "prepare-rewrite",
        help="plan new rewrites until the accepted target is reached",
    )
    rewrite.add_argument("--sources", type=Path, required=True)
    rewrite.add_argument("--tasks-dir", type=Path, required=True)
    rewrite.add_argument("--existing-requests", type=Path)
    rewrite.add_argument("--existing-results", type=Path)
    rewrite.add_argument("--target-count", type=int, default=100)
    rewrite.add_argument("--styles", nargs="+")
    rewrite.add_argument("--seed", type=int, default=0)
    rewrite.add_argument("--output", type=Path, required=True)
    _add_sampling_arguments(rewrite, temperature=0.7, max_tokens=8192)
    rewrite.set_defaults(handler=prepare_rewrite)

    export = subparsers.add_parser("gemini-export", help="write Gemini batch JSONL")
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.set_defaults(handler=gemini_export)

    import_parser = subparsers.add_parser(
        "gemini-import",
        help="normalize downloaded Gemini batch results",
    )
    import_parser.add_argument("--input", type=Path, required=True)
    import_parser.add_argument("--requests", type=Path, required=True)
    import_parser.add_argument("--model", required=True)
    import_parser.add_argument("--output", type=Path, required=True)
    import_parser.set_defaults(handler=gemini_import)

    submit = subparsers.add_parser("gemini-submit", help="submit a Gemini batch")
    submit.add_argument("--input", type=Path, required=True)
    submit.add_argument("--model", required=True)
    submit.add_argument("--display-name", required=True)
    submit.set_defaults(handler=gemini_submit)

    status = subparsers.add_parser("gemini-status", help="show Gemini batch state")
    status.add_argument("--batch-name", required=True)
    status.set_defaults(handler=gemini_status)

    download = subparsers.add_parser(
        "gemini-download",
        help="download a completed Gemini batch",
    )
    download.add_argument("--batch-name", required=True)
    download.add_argument("--output", type=Path, required=True)
    download.set_defaults(handler=gemini_download)

    local = subparsers.add_parser("vllm-run", help="run requests with local vLLM")
    local.add_argument("--input", type=Path, required=True)
    local.add_argument("--output", type=Path, required=True)
    local.add_argument("--model", required=True)
    local.add_argument("--tensor-parallel-size", type=int, default=1)
    local.add_argument("--trust-remote-code", action="store_true")
    local.set_defaults(handler=vllm_run)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
