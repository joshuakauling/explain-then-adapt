"""Local vLLM execution for provider-independent generation requests."""

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Mapping, Sequence, Tuple

from ..records import GenerationRequest, GenerationResult, SamplingParameters, TokenUsage


def to_vllm_messages(request: GenerationRequest) -> List[Dict[str, str]]:
    """Convert an internal request to vLLM's OpenAI-style chat messages."""
    return [message.to_dict() for message in request.messages]


def _sampling_key(parameters: SamplingParameters) -> Tuple[float, float, int, int]:
    return (
        parameters.temperature,
        parameters.top_p,
        parameters.max_tokens,
        parameters.seed if parameters.seed is not None else -1,
    )


def run_vllm_requests(
    requests: Sequence[GenerationRequest],
    *,
    model: str,
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = False,
) -> List[GenerationResult]:
    """Execute requests locally, grouping calls by sampling configuration."""
    if not requests:
        return []
    if any(request.sampling.use_provider_defaults for request in requests):
        raise ValueError(
            "vLLM requests require explicit sampling parameters; provider defaults "
            "are only supported by provider adapters such as Gemini."
        )
    try:
            from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "Local execution requires the optional 'vllm' dependencies. "
            "Install the project with: pip install -e '.[vllm]'"
        ) from error

    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=trust_remote_code,
    )
    grouped: DefaultDict[
        Tuple[float, float, int, int], List[Tuple[int, GenerationRequest]]
    ] = defaultdict(list)
    for index, request in enumerate(requests):
        grouped[_sampling_key(request.sampling)].append((index, request))

    indexed_results: Dict[int, GenerationResult] = {}
    for group in grouped.values():
        parameters = group[0][1].sampling
        sampling_kwargs: Dict[str, Any] = {
            "temperature": parameters.temperature,
            "top_p": parameters.top_p,
            "max_tokens": parameters.max_tokens,
        }
        if parameters.seed is not None:
            sampling_kwargs["seed"] = parameters.seed
        outputs = llm.chat(
            [to_vllm_messages(request) for _, request in group],
            sampling_params=SamplingParams(**sampling_kwargs),
            use_tqdm=True,
        )
        if len(outputs) != len(group):
            raise RuntimeError("vLLM returned a different number of outputs than requests.")

        for (index, request), request_output in zip(group, outputs):
            candidates = getattr(request_output, "outputs", ())
            if not candidates:
                result = GenerationResult(
                    request_id=request.request_id,
                    task_id=request.task_id,
                    stage=request.stage,
                    backend="vllm",
                    model=model,
                    prompt_version=request.prompt_version,
                    hint_mode=request.hint_mode,
                    few_shot_task_ids=request.few_shot_task_ids,
                    sampling=request.sampling,
                    augmentation=request.augmentation,
                    error="vLLM returned no candidate output.",
                    metadata=dict(request.metadata),
                )
            else:
                candidate = candidates[0]
                prompt_token_ids = getattr(request_output, "prompt_token_ids", ()) or ()
                output_token_ids = getattr(candidate, "token_ids", ()) or ()
                result = GenerationResult(
                    request_id=request.request_id,
                    task_id=request.task_id,
                    stage=request.stage,
                    backend="vllm",
                    model=model,
                    prompt_version=request.prompt_version,
                    hint_mode=request.hint_mode,
                    few_shot_task_ids=request.few_shot_task_ids,
                    sampling=request.sampling,
                    augmentation=request.augmentation,
                    raw_output=str(getattr(candidate, "text", "")),
                    finish_reason=(
                        str(getattr(candidate, "finish_reason"))
                        if getattr(candidate, "finish_reason", None) is not None
                        else None
                    ),
                    usage=TokenUsage(
                        prompt_tokens=len(prompt_token_ids),
                        completion_tokens=len(output_token_ids),
                    ),
                    metadata=dict(request.metadata),
                )
            indexed_results[index] = result

    return [indexed_results[index] for index in range(len(requests))]
