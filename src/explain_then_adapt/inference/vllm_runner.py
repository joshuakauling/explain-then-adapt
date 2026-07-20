"""vLLM execution for Reasoning and Prediction Model inference."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from explain_then_adapt.arc.io import load_task
from explain_then_adapt.training.ttt_data import TTTAugmentation, normalize_ttt_task_ids

from .artifacts import (
    JsonlArtifactWriter,
    chunked,
    manifest_path_for,
    sha256_file,
    task_sources_sha256,
)
from .config import GUIDANCE_MODES, InferenceConfig
from .planning import (
    PredictionRequest,
    ReasoningRequest,
    build_prediction_requests,
    build_reasoning_requests,
    candidate_count_by_test_input,
    validate_protocol_arguments,
)
from .prompts import (
    build_prediction_prompt_from_task,
    build_reasoning_prompt_from_task,
    extract_guidance,
    load_guidance,
    prompt_sha256,
    require_guidance_for_variants,
)


def _vllm_types() -> Tuple[Any, Any, Any]:
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as error:
        raise RuntimeError(
            "install the optional inference dependencies with "
            "`python -m pip install -e '.[vllm]'`."
        ) from error
    return LLM, SamplingParams, LoRARequest


def _resolved_engine_value(default: Any, override: Optional[Any]) -> Any:
    return default if override is None else override


def _validate_engine_overrides(
    *,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    request_batch_size: int,
) -> None:
    if isinstance(tensor_parallel_size, bool) or tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be a positive integer.")
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1].")
    if isinstance(request_batch_size, bool) or request_batch_size <= 0:
        raise ValueError("request_batch_size must be a positive integer.")


def _preflight_output(output_path: Path) -> None:
    manifest_path = manifest_path_for(output_path)
    partial_path = output_path.with_name(f".{output_path.name}.partial")
    for path in (output_path, manifest_path, partial_path):
        if path.exists():
            raise FileExistsError(
                f"inference artifact already exists and will not be overwritten: "
                f"{path}."
            )


def _token_count(value: Any) -> Optional[int]:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _encoded_token_count(llm: Any, text: str) -> int:
    tokenizer = llm.get_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens.tolist() if hasattr(tokens, "tolist") else tokens)


def _request_prompt_token_count(output: Any, llm: Any) -> int:
    count = _token_count(getattr(output, "prompt_token_ids", None))
    if count is not None:
        return count
    rendered_prompt = getattr(output, "prompt", None)
    if not isinstance(rendered_prompt, str):
        raise ValueError("vLLM output contains neither prompt tokens nor prompt text.")
    return _encoded_token_count(llm, rendered_prompt)


def _completion_record(
    completion: Any,
    *,
    sample_index: int,
    llm: Any,
) -> Dict[str, Any]:
    text = getattr(completion, "text", None)
    if not isinstance(text, str):
        raise ValueError("vLLM completion has no text output.")
    token_count = _token_count(getattr(completion, "token_ids", None))
    if token_count is None:
        token_count = _encoded_token_count(llm, text)
    finish_reason = getattr(completion, "finish_reason", None)
    stop_reason = getattr(completion, "stop_reason", None)
    return {
        "sample_index": sample_index,
        "text": text,
        "generated_token_count": token_count,
        "finish_reason": (str(finish_reason) if finish_reason is not None else None),
        "stop_reason": str(stop_reason) if stop_reason is not None else None,
    }


def _source_path_metadata(path: Optional[Path]) -> Dict[str, Optional[str]]:
    if path is None:
        return {"path": None, "sha256": None}
    return {"path": str(path), "sha256": sha256_file(path)}


def run_reasoning_inference(
    *,
    config: InferenceConfig,
    task_ids: Sequence[str],
    tasks_directory: Path,
    protocol: str,
    model: str,
    output_path: Path,
    guidance_budget: Optional[int] = None,
    augmentation_plan: Optional[Mapping[str, Sequence[TTTAugmentation]]] = None,
    augmentation_plan_path: Optional[Path] = None,
    tensor_parallel_size: Optional[int] = None,
    gpu_memory_utilization: Optional[float] = None,
    request_batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Sample exactly one RM trace per required original or augmented view."""
    resolved_budget = validate_protocol_arguments(protocol, guidance_budget)
    normalized_ids = normalize_ttt_task_ids(task_ids)
    requests = build_reasoning_requests(
        config=config,
        task_ids=normalized_ids,
        protocol=protocol,
        guidance_budget=resolved_budget,
        augmentation_plan=augmentation_plan,
    )
    _preflight_output(output_path)

    engine = config.reasoning.engine
    resolved_tp = int(
        _resolved_engine_value(engine.tensor_parallel_size, tensor_parallel_size)
    )
    resolved_gpu_memory = float(
        _resolved_engine_value(
            engine.gpu_memory_utilization,
            gpu_memory_utilization,
        )
    )
    resolved_batch_size = int(
        _resolved_engine_value(engine.request_batch_size, request_batch_size)
    )
    _validate_engine_overrides(
        tensor_parallel_size=resolved_tp,
        gpu_memory_utilization=resolved_gpu_memory,
        request_batch_size=resolved_batch_size,
    )

    LLM, SamplingParams, _ = _vllm_types()
    llm = LLM(
        model=model,
        dtype=engine.dtype,
        enable_lora=False,
        gpu_memory_utilization=resolved_gpu_memory,
        max_model_len=engine.max_model_len,
        tensor_parallel_size=resolved_tp,
        seed=config.sampling_seed,
        generation_config="vllm",
    )
    sampling = config.reasoning.sampling
    sampling_params = SamplingParams(
        n=1,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        max_tokens=sampling.max_tokens,
    )

    task_cache = {
        task_id: load_task(task_id, tasks_directory) for task_id in normalized_ids
    }
    valid_count = 0
    invalid_ids: List[str] = []
    total_prompt_tokens = 0
    total_generated_tokens = 0

    with JsonlArtifactWriter(output_path) as writer:
        for request_batch in chunked(requests, resolved_batch_size):
            prompts = [
                build_reasoning_prompt_from_task(
                    request,
                    task_cache[request.variant.task_id],
                    config,
                )
                for request in request_batch
            ]
            conversations = [
                [{"role": "user", "content": prompt}] for prompt in prompts
            ]
            outputs = llm.chat(
                conversations,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            if len(outputs) != len(request_batch):
                raise RuntimeError("vLLM returned the wrong number of RM outputs.")

            for request, prompt, output in zip(request_batch, prompts, outputs):
                completions = list(getattr(output, "outputs", ()))
                if len(completions) != 1:
                    raise RuntimeError(
                        f"RM request {request.request_id!r} returned "
                        f"{len(completions)} samples; exactly one is required."
                    )
                completion = _completion_record(
                    completions[0],
                    sample_index=0,
                    llm=llm,
                )
                try:
                    guidance = extract_guidance(str(completion["text"]))
                    validation_error = None
                    valid_count += 1
                except (TypeError, ValueError) as error:
                    guidance = None
                    validation_error = str(error)
                    invalid_ids.append(request.request_id)

                prompt_tokens = _request_prompt_token_count(output, llm)
                total_prompt_tokens += prompt_tokens
                total_generated_tokens += int(completion["generated_token_count"])
                writer.write(
                    {
                        "schema_version": 1,
                        "kind": "reasoning_guidance",
                        "request_id": request.request_id,
                        "task_id": request.variant.task_id,
                        "variant": request.variant.to_dict(),
                        "user_prompt_sha256": prompt_sha256(prompt),
                        "prompt_token_count": prompt_tokens,
                        "raw_output": completion["text"],
                        "generated_token_count": completion["generated_token_count"],
                        "finish_reason": completion["finish_reason"],
                        "stop_reason": completion["stop_reason"],
                        "guidance": guidance,
                        "validation_error": validation_error,
                    }
                )

        manifest = {
            "schema_version": 1,
            "kind": "reasoning_inference_run",
            "protocol": protocol,
            "guidance_budget": resolved_budget,
            "rm_samples_per_request": 1,
            "model": model,
            "task_count": len(normalized_ids),
            "task_ids": normalized_ids,
            "task_sources_sha256": task_sources_sha256(
                tasks_directory,
                normalized_ids,
            ),
            "augmentation_plan": _source_path_metadata(augmentation_plan_path),
            "request_count": len(requests),
            "valid_guidance_count": valid_count,
            "invalid_guidance_count": len(invalid_ids),
            "invalid_request_ids": invalid_ids,
            "total_prompt_tokens": total_prompt_tokens,
            "total_generated_tokens": total_generated_tokens,
            "engine": {
                "dtype": engine.dtype,
                "gpu_memory_utilization": resolved_gpu_memory,
                "max_model_len": engine.max_model_len,
                "tensor_parallel_size": resolved_tp,
                "request_batch_size": resolved_batch_size,
            },
            "sampling": {
                "seed": config.sampling_seed,
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "max_tokens": sampling.max_tokens,
            },
            "resolved_config": config.to_dict(),
        }
        writer.complete(manifest)

    if invalid_ids:
        raise RuntimeError(
            f"RM produced {len(invalid_ids)} malformed guidance traces; the raw "
            f"outputs were preserved in {output_path}."
        )
    return manifest


def _adapter_complete(path: Path, task_id: str) -> bool:
    manifest_path = path / "ttt_manifest.json"
    if (
        not path.is_dir()
        or not (path / "adapter_config.json").is_file()
        or not manifest_path.is_file()
    ):
        return False
    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False
    valid_manifest = (
        isinstance(manifest, Mapping)
        and manifest.get("schema_version") == 1
        and manifest.get("kind") == "ttt_task_adapter"
        and manifest.get("task_id") == task_id
        and manifest.get("optimizer_updates") == 64
    )
    return valid_manifest and any(
        (path / name).is_file() and (path / name).stat().st_size > 0
        for name in (
            "adapter_model.safetensors",
            "adapter_model.bin",
            "adapter_model.pt",
        )
    )


def _validate_adapters(adapter_root: Path, task_ids: Sequence[str]) -> None:
    if not adapter_root.is_dir():
        raise FileNotFoundError(f"TTT adapter root does not exist: {adapter_root}.")
    missing = [
        task_id
        for task_id in task_ids
        if not _adapter_complete(adapter_root / task_id, task_id)
    ]
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} task adapters are missing or incomplete under "
            f"{adapter_root}; first tasks: {preview}."
        )


def _prediction_record(
    *,
    request: PredictionRequest,
    prompt: str,
    output: Any,
    llm: Any,
    protocol: str,
    guidance_mode: str,
    adapter_used: bool,
) -> Tuple[Dict[str, Any], int, int]:
    completions = list(getattr(output, "outputs", ()))
    if len(completions) != request.sample_count:
        raise RuntimeError(
            f"PM request {request.request_id!r} returned {len(completions)} "
            f"samples; expected {request.sample_count}."
        )
    output_records = [
        _completion_record(value, sample_index=index, llm=llm)
        for index, value in enumerate(completions)
    ]
    prompt_tokens = _request_prompt_token_count(output, llm)
    generated_tokens = sum(
        int(value["generated_token_count"]) for value in output_records
    )
    return (
        {
            "schema_version": 1,
            "kind": "prediction_candidates",
            "request_id": request.request_id,
            "task_id": request.variant.task_id,
            "test_index": request.test_index,
            "variant": request.variant.to_dict(),
            "protocol": protocol,
            "guidance_mode": guidance_mode,
            "guidance_key": (
                request.variant.key if guidance_mode == "guided" else None
            ),
            "adapter_used": adapter_used,
            "prompt_sha256": prompt_sha256(prompt),
            "prompt_token_count": prompt_tokens,
            "sample_count": request.sample_count,
            "outputs": output_records,
        },
        prompt_tokens,
        generated_tokens,
    )


def run_prediction_inference(
    *,
    config: InferenceConfig,
    task_ids: Sequence[str],
    tasks_directory: Path,
    protocol: str,
    guidance_mode: str,
    model: str,
    output_path: Path,
    guidance_budget: Optional[int] = None,
    augmentation_plan: Optional[Mapping[str, Sequence[TTTAugmentation]]] = None,
    augmentation_plan_path: Optional[Path] = None,
    guidance_path: Optional[Path] = None,
    ttt_adapter_root: Optional[Path] = None,
    tensor_parallel_size: Optional[int] = None,
    gpu_memory_utilization: Optional[float] = None,
    request_batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate raw PM candidates, optionally through per-task TTT adapters."""
    resolved_budget = validate_protocol_arguments(protocol, guidance_budget)
    if guidance_mode not in GUIDANCE_MODES:
        raise ValueError(f"guidance_mode must be one of {GUIDANCE_MODES}.")
    if guidance_mode == "guided" and guidance_path is None:
        raise ValueError("guided prediction requires a guidance artifact.")
    if guidance_mode == "unguided" and guidance_path is not None:
        raise ValueError("unguided prediction must not receive a guidance artifact.")
    if protocol == "budgeted64" and guidance_mode != "guided":
        raise ValueError(
            "budgeted64 is the guided thesis ablation; use augmented64 for the "
            "unguided 64-variant reference."
        )
    if protocol == "budgeted64" and ttt_adapter_root is None:
        raise ValueError("budgeted64 requires task-specific TTT adapters.")

    normalized_ids = normalize_ttt_task_ids(task_ids)
    requests = build_prediction_requests(
        config=config,
        task_ids=normalized_ids,
        tasks_directory=tasks_directory,
        protocol=protocol,
        guidance_budget=resolved_budget,
        augmentation_plan=augmentation_plan,
    )
    variants = [request.variant for request in requests]
    guidance_by_key: Mapping[str, str] = {}
    if guidance_path is not None:
        guidance_by_key = load_guidance(guidance_path)
        require_guidance_for_variants(variants, guidance_by_key)
    if ttt_adapter_root is not None:
        _validate_adapters(ttt_adapter_root, normalized_ids)
    _preflight_output(output_path)

    engine = config.prediction.engine
    resolved_tp = int(
        _resolved_engine_value(engine.tensor_parallel_size, tensor_parallel_size)
    )
    resolved_gpu_memory = float(
        _resolved_engine_value(
            engine.gpu_memory_utilization,
            gpu_memory_utilization,
        )
    )
    resolved_batch_size = int(
        _resolved_engine_value(engine.request_batch_size, request_batch_size)
    )
    _validate_engine_overrides(
        tensor_parallel_size=resolved_tp,
        gpu_memory_utilization=resolved_gpu_memory,
        request_batch_size=resolved_batch_size,
    )

    LLM, SamplingParams, LoRARequest = _vllm_types()
    use_ttt = ttt_adapter_root is not None
    max_model_len = engine.ttt_max_model_len if use_ttt else engine.max_model_len
    llm_kwargs: Dict[str, Any] = {
        "model": model,
        "dtype": engine.dtype,
        "enable_lora": use_ttt,
        "gpu_memory_utilization": resolved_gpu_memory,
        "max_model_len": max_model_len,
        "tensor_parallel_size": resolved_tp,
        "seed": config.sampling_seed,
        "generation_config": "vllm",
    }
    if use_ttt:
        llm_kwargs["max_lora_rank"] = engine.max_lora_rank
    llm = LLM(**llm_kwargs)

    sampling = config.prediction.sampling
    sample_counts = {request.sample_count for request in requests}
    if len(sample_counts) != 1:
        raise RuntimeError("one inference run must use one PM sample count.")
    samples_per_prompt = next(iter(sample_counts))
    sampling_params = SamplingParams(
        n=samples_per_prompt,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        max_tokens=sampling.max_tokens,
        stop=list(config.prediction.stop),
    )

    task_cache = {
        task_id: load_task(task_id, tasks_directory) for task_id in normalized_ids
    }
    total_prompt_tokens = 0
    total_generated_tokens = 0
    total_candidates = 0

    groups: Dict[str, List[PredictionRequest]] = defaultdict(list)
    if use_ttt:
        for request in requests:
            groups[request.variant.task_id].append(request)
    else:
        groups["__base__"] = list(requests)

    with JsonlArtifactWriter(output_path) as writer:
        for adapter_index, group_key in enumerate(sorted(groups), start=1):
            lora_request = None
            if use_ttt:
                assert ttt_adapter_root is not None
                lora_request = LoRARequest(
                    lora_name=group_key,
                    lora_int_id=adapter_index,
                    lora_path=str(ttt_adapter_root / group_key),
                )
            for request_batch in chunked(groups[group_key], resolved_batch_size):
                prompts = []
                for request in request_batch:
                    guidance = (
                        guidance_by_key[request.variant.key]
                        if guidance_mode == "guided"
                        else None
                    )
                    prompts.append(
                        build_prediction_prompt_from_task(
                            request,
                            task_cache[request.variant.task_id],
                            config,
                            guidance=guidance,
                        )
                    )
                kwargs: Dict[str, Any] = {
                    "sampling_params": sampling_params,
                    "use_tqdm": False,
                }
                if lora_request is not None:
                    kwargs["lora_request"] = lora_request
                outputs = llm.generate(prompts, **kwargs)
                if len(outputs) != len(request_batch):
                    raise RuntimeError("vLLM returned the wrong number of PM outputs.")

                for request, prompt, output in zip(
                    request_batch,
                    prompts,
                    outputs,
                ):
                    record, prompt_tokens, generated_tokens = _prediction_record(
                        request=request,
                        prompt=prompt,
                        output=output,
                        llm=llm,
                        protocol=protocol,
                        guidance_mode=guidance_mode,
                        adapter_used=use_ttt,
                    )
                    writer.write(record)
                    total_prompt_tokens += prompt_tokens
                    total_generated_tokens += generated_tokens
                    total_candidates += request.sample_count

        candidates_by_test = set(candidate_count_by_test_input(requests).values())
        if len(candidates_by_test) != 1:
            raise RuntimeError("candidate budgets differ between test inputs.")
        manifest = {
            "schema_version": 1,
            "kind": "prediction_inference_run",
            "protocol": protocol,
            "guidance_mode": guidance_mode,
            "guidance_budget": resolved_budget,
            "model": model,
            "task_count": len(normalized_ids),
            "task_ids": normalized_ids,
            "task_sources_sha256": task_sources_sha256(
                tasks_directory,
                normalized_ids,
            ),
            "augmentation_plan": _source_path_metadata(augmentation_plan_path),
            "guidance_artifact": _source_path_metadata(guidance_path),
            "ttt_enabled": use_ttt,
            "ttt_adapter_root": (
                str(ttt_adapter_root) if ttt_adapter_root is not None else None
            ),
            "request_count": len(requests),
            "samples_per_prompt": samples_per_prompt,
            "candidates_per_test_input": next(iter(candidates_by_test)),
            "total_candidates": total_candidates,
            "total_prompt_tokens": total_prompt_tokens,
            "total_generated_tokens": total_generated_tokens,
            "outputs_are_in_variant_space": True,
            "engine": {
                "dtype": engine.dtype,
                "gpu_memory_utilization": resolved_gpu_memory,
                "max_model_len": max_model_len,
                "tensor_parallel_size": resolved_tp,
                "request_batch_size": resolved_batch_size,
                "max_lora_rank": engine.max_lora_rank if use_ttt else None,
            },
            "sampling": {
                "seed": config.sampling_seed,
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "max_tokens": sampling.max_tokens,
                "stop": list(config.prediction.stop),
            },
            "resolved_config": config.to_dict(),
        }
        writer.complete(manifest)
    return manifest
