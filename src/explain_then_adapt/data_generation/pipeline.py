"""Provider-independent construction and normalization of pipeline requests."""

from dataclasses import replace
from typing import List, Optional, Sequence, Tuple

from explain_then_adapt.arc.types import Example

from .augmentation import apply_augmentation
from .hints import Hint
from .postprocessing import normalize_trace
from .prompts import (
    INITIAL_PROMPT_VERSION,
    JUDGE_PROMPT_VERSION,
    REWRITE_PROMPT_VERSION,
    FewShotExample,
    build_initial_messages,
    build_judge_messages,
    build_rewrite_messages,
)
from .records import (
    AugmentationSpec,
    GenerationRequest,
    GenerationResult,
    GenerationStage,
    HintMode,
    SamplingParameters,
    make_request_id,
)
from .validation import JUDGE_VOTE_COUNT, validate_trace_format


def build_initial_request(
    *,
    task_id: str,
    puzzle: List[Example],
    hint: Optional[Hint],
    few_shots: Tuple[FewShotExample, ...],
    sampling: Optional[SamplingParameters] = None,
    candidate_index: int = 0,
) -> GenerationRequest:
    """Construct one reproducibly identified initial-generation request."""
    parameters = sampling or SamplingParameters(temperature=0.7)
    few_shot_ids = tuple(example.task_id for example in few_shots)
    hint_mode = HintMode.PROVIDED if hint is not None else HintMode.NONE
    identity = {
        "prompt_version": INITIAL_PROMPT_VERSION,
        "few_shot_task_ids": list(few_shot_ids),
        "hint_mode": hint_mode.value,
        "candidate_index": candidate_index,
        "sampling": parameters.to_dict(),
    }
    return GenerationRequest(
        request_id=make_request_id(GenerationStage.INITIAL, task_id, identity),
        task_id=task_id,
        stage=GenerationStage.INITIAL,
        messages=build_initial_messages(
            puzzle,
            hint=hint,
            few_shots=few_shots,
        ),
        prompt_version=INITIAL_PROMPT_VERSION,
        hint_mode=hint_mode,
        few_shot_task_ids=few_shot_ids,
        sampling=parameters,
        metadata={"candidate_index": candidate_index},
    )


def build_judge_requests(
    *,
    task_id: str,
    puzzle: List[Example],
    candidate_trace: str,
    source_request_id: str,
    hint: Optional[Hint],
    sampling: Optional[SamplingParameters] = None,
) -> List[GenerationRequest]:
    """Construct the five independent requests required for automatic acceptance."""
    parameters = sampling or SamplingParameters(temperature=0.7, max_tokens=2048)
    hint_mode = HintMode.PROVIDED if hint is not None else HintMode.NONE
    requests: List[GenerationRequest] = []
    for judge_index in range(JUDGE_VOTE_COUNT):
        judge_parameters = (
            replace(parameters, seed=parameters.seed + judge_index)
            if parameters.seed is not None
            else parameters
        )
        identity = {
            "prompt_version": JUDGE_PROMPT_VERSION,
            "source_request_id": source_request_id,
            "judge_index": judge_index,
            "sampling": judge_parameters.to_dict(),
        }
        requests.append(
            GenerationRequest(
                request_id=make_request_id(GenerationStage.JUDGE, task_id, identity),
                task_id=task_id,
                stage=GenerationStage.JUDGE,
                messages=build_judge_messages(
                    puzzle,
                    candidate_trace,
                    hint=hint,
                ),
                prompt_version=JUDGE_PROMPT_VERSION,
                hint_mode=hint_mode,
                sampling=judge_parameters,
                metadata={
                    "source_request_id": source_request_id,
                    "judge_index": judge_index,
                },
            )
        )
    return requests


def build_rewrite_request(
    *,
    task_id: str,
    puzzle: List[Example],
    accepted_trace: str,
    spec: AugmentationSpec,
    sampling: Optional[SamplingParameters] = None,
) -> GenerationRequest:
    """Construct one trace-rewrite request and its transformed demonstration pairs."""
    parameters = sampling or SamplingParameters(temperature=0.7)
    transformed_puzzle = apply_augmentation(puzzle, spec)
    identity = {
        "prompt_version": REWRITE_PROMPT_VERSION,
        "augmentation": spec.to_dict(),
        "sampling": parameters.to_dict(),
    }
    return GenerationRequest(
        request_id=make_request_id(GenerationStage.REWRITE, task_id, identity),
        task_id=task_id,
        stage=GenerationStage.REWRITE,
        messages=build_rewrite_messages(
            puzzle,
            accepted_trace,
            transformed_puzzle,
            spec,
        ),
        prompt_version=REWRITE_PROMPT_VERSION,
        sampling=parameters,
        augmentation=spec,
        metadata={"source_trace_id": spec.source_trace_id},
    )


def apply_static_validation(result: GenerationResult) -> GenerationResult:
    """Normalize a trace result and attach schema-validation provenance."""
    static_result = validate_trace_format(result.raw_output)
    return replace(
        result,
        normalized_output=static_result.normalized_text,
        validation={"static": dict(static_result.to_dict())},
    )


def normalize_judge_output(result: GenerationResult) -> GenerationResult:
    """Trim a judge response without applying reasoning-trace normalization."""
    return replace(result, normalized_output=normalize_trace(result.raw_output))
