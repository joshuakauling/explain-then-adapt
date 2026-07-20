"""Deterministic request planning for the three thesis inference protocols."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from explain_then_adapt.arc.augmented_keys import make_augmented_key
from explain_then_adapt.arc.io import load_task
from explain_then_adapt.training.ttt_data import (
    TTTAugmentation,
    generate_ttt_augmentation_plan,
    load_ttt_augmentation_plan,
    normalize_ttt_task_ids,
    selected_guidance_variant_indices,
    ttt_augmentation_plan_payload,
)

from .config import GUIDANCE_BUDGETS, INFERENCE_PROTOCOLS, InferenceConfig

IDENTITY_VALUE_MAPPING = "0123456789"


@dataclass(frozen=True)
class InferenceVariant:
    """One original or augmented view of an ARC task."""

    task_id: str
    transformation_code: str = "ID"
    value_mapping: str = IDENTITY_VALUE_MAPPING
    order_mapping: Optional[str] = None
    variant_index: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.task_id or "_" in self.task_id:
            raise ValueError("task_id must be non-empty and contain no underscore.")
        if self.variant_index is None:
            if (
                self.transformation_code != "ID"
                or self.value_mapping != IDENTITY_VALUE_MAPPING
                or self.order_mapping is not None
            ):
                raise ValueError(
                    "an original inference variant must use the identity transform."
                )
            return
        if isinstance(self.variant_index, bool) or self.variant_index < 0:
            raise ValueError("variant_index must be a non-negative integer.")
        if self.order_mapping is None:
            raise ValueError("an augmented inference variant needs order_mapping.")
        make_augmented_key(
            self.task_id,
            self.transformation_code,
            self.value_mapping,
            self.order_mapping,
        )

    @property
    def is_augmented(self) -> bool:
        return self.variant_index is not None

    @property
    def key(self) -> str:
        if not self.is_augmented:
            return self.task_id
        assert self.order_mapping is not None
        return make_augmented_key(
            self.task_id,
            self.transformation_code,
            self.value_mapping,
            self.order_mapping,
        )

    @classmethod
    def from_ttt(cls, value: TTTAugmentation) -> "InferenceVariant":
        return cls(
            task_id=value.task_id,
            transformation_code=value.transformation_code,
            value_mapping=value.value_mapping,
            order_mapping=value.order_mapping,
            variant_index=value.variant_index,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "task_id": self.task_id,
            "is_augmented": self.is_augmented,
            "transformation_code": self.transformation_code,
            "value_mapping": self.value_mapping,
            "order_mapping": self.order_mapping,
            "variant_index": self.variant_index,
        }


@dataclass(frozen=True)
class ReasoningRequest:
    """One RM prompt; the final protocol always draws exactly one sample."""

    request_id: str
    variant: InferenceVariant

    def __post_init__(self) -> None:
        if self.request_id != self.variant.key:
            raise ValueError("reasoning request_id must equal its variant key.")


@dataclass(frozen=True)
class PredictionRequest:
    """One PM prompt and its required number of samples."""

    request_id: str
    variant: InferenceVariant
    test_index: int
    sample_count: int

    def __post_init__(self) -> None:
        expected_id = f"{self.variant.key}__{self.test_index}"
        if self.request_id != expected_id:
            raise ValueError(f"prediction request_id must be {expected_id!r}.")
        if isinstance(self.test_index, bool) or self.test_index < 0:
            raise ValueError("test_index must be a non-negative integer.")
        if isinstance(self.sample_count, bool) or self.sample_count <= 0:
            raise ValueError("sample_count must be a positive integer.")


def validate_protocol_arguments(
    protocol: str,
    guidance_budget: Optional[int],
) -> Optional[int]:
    """Validate when a guidance budget is required or forbidden."""
    if protocol not in INFERENCE_PROTOCOLS:
        raise ValueError(
            f"unknown inference protocol {protocol!r}; expected one of "
            f"{INFERENCE_PROTOCOLS}."
        )
    if protocol == "budgeted64":
        if guidance_budget is None:
            raise ValueError("budgeted64 requires a guidance budget.")
        if isinstance(guidance_budget, bool) or guidance_budget not in GUIDANCE_BUDGETS:
            raise ValueError(f"guidance budget must be one of {GUIDANCE_BUDGETS}.")
        return guidance_budget
    if guidance_budget is not None:
        raise ValueError(
            f"{protocol} does not accept a guidance budget; use budgeted64."
        )
    return None


def protocol_needs_augmentation_plan(
    protocol: str,
    guidance_budget: Optional[int],
) -> bool:
    resolved_budget = validate_protocol_arguments(protocol, guidance_budget)
    return protocol == "augmented64" or (
        protocol == "budgeted64" and resolved_budget != 0
    )


def create_augmentation_plan(
    *,
    config: InferenceConfig,
    task_ids: Sequence[str],
    tasks_directory: Path,
) -> Dict[str, List[TTTAugmentation]]:
    """Create the canonical 64-variant plan shared by RM, TTT, and PM."""
    return generate_ttt_augmentation_plan(
        task_ids=task_ids,
        tasks_directory=tasks_directory,
        seed=config.augmentation_seed,
        variants_per_transform=config.variants_per_transform,
    )


def save_augmentation_plan(
    *,
    config: InferenceConfig,
    plan: Mapping[str, Sequence[TTTAugmentation]],
    output_path: Path,
) -> None:
    """Persist a canonical plan without overwriting an existing artifact."""
    if output_path.exists():
        raise FileExistsError(f"augmentation plan already exists: {output_path}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = ttt_augmentation_plan_payload(
        plan,
        seed=config.augmentation_seed,
        variants_per_transform=config.variants_per_transform,
    )
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary plan already exists: {temporary}.")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(output_path)


def load_augmentation_plan(
    *,
    config: InferenceConfig,
    path: Path,
    task_ids: Sequence[str],
    tasks_directory: Path,
) -> Dict[str, List[TTTAugmentation]]:
    """Load and validate the same plan format consumed by online TTT."""
    return load_ttt_augmentation_plan(
        path,
        task_ids=task_ids,
        tasks_directory=tasks_directory,
        variants_per_transform=config.variants_per_transform,
    )


def _variants_for_protocol(
    *,
    task_ids: Sequence[str],
    protocol: str,
    guidance_budget: Optional[int],
    augmentation_plan: Optional[Mapping[str, Sequence[TTTAugmentation]]],
    variants_per_transform: int,
) -> List[InferenceVariant]:
    resolved_budget = validate_protocol_arguments(protocol, guidance_budget)
    normalized_ids = normalize_ttt_task_ids(task_ids)
    needs_plan = protocol == "augmented64" or (
        protocol == "budgeted64" and resolved_budget != 0
    )
    if needs_plan and augmentation_plan is None:
        raise ValueError(f"{protocol} requires an augmentation plan.")
    if augmentation_plan is not None and set(augmentation_plan) != set(normalized_ids):
        raise ValueError("augmentation plan task IDs do not match the selection.")

    variants: List[InferenceVariant] = []
    selected_indices: Optional[set] = None
    if protocol == "budgeted64" and resolved_budget:
        selected_indices = set(
            selected_guidance_variant_indices(
                resolved_budget,
                variants_per_transform=variants_per_transform,
            )
        )

    for task_id in normalized_ids:
        if protocol == "standard32" or (
            protocol == "budgeted64" and resolved_budget == 0
        ):
            variants.append(InferenceVariant(task_id=task_id))
            continue
        assert augmentation_plan is not None
        task_variants = list(augmentation_plan[task_id])
        if selected_indices is not None:
            task_variants = [
                value
                for value in task_variants
                if value.variant_index in selected_indices
            ]
        variants.extend(InferenceVariant.from_ttt(value) for value in task_variants)

    expected_per_task = 1
    if protocol == "augmented64":
        expected_per_task = 64
    elif protocol == "budgeted64" and resolved_budget:
        expected_per_task = resolved_budget
    if len(variants) != len(normalized_ids) * expected_per_task:
        raise ValueError(
            f"{protocol} resolved to {len(variants)} variants; expected "
            f"{len(normalized_ids) * expected_per_task}."
        )
    return variants


def build_reasoning_requests(
    *,
    config: InferenceConfig,
    task_ids: Sequence[str],
    protocol: str,
    guidance_budget: Optional[int] = None,
    augmentation_plan: Optional[Mapping[str, Sequence[TTTAugmentation]]] = None,
) -> List[ReasoningRequest]:
    """Plan one and only one RM sample for every required task view."""
    variants = _variants_for_protocol(
        task_ids=task_ids,
        protocol=protocol,
        guidance_budget=guidance_budget,
        augmentation_plan=augmentation_plan,
        variants_per_transform=config.variants_per_transform,
    )
    return [
        ReasoningRequest(request_id=variant.key, variant=variant)
        for variant in variants
    ]


def _sample_count(
    *,
    config: InferenceConfig,
    protocol: str,
    guidance_budget: Optional[int],
) -> int:
    resolved_budget = validate_protocol_arguments(protocol, guidance_budget)
    if protocol == "standard32":
        return config.protocols.standard_samples
    if protocol == "augmented64":
        return config.protocols.augmented_samples_per_variant
    assert resolved_budget is not None
    if resolved_budget == 0:
        return config.protocols.budgeted_total_samples
    return config.protocols.budgeted_total_samples // resolved_budget


def build_prediction_requests(
    *,
    config: InferenceConfig,
    task_ids: Sequence[str],
    tasks_directory: Path,
    protocol: str,
    guidance_budget: Optional[int] = None,
    augmentation_plan: Optional[Mapping[str, Sequence[TTTAugmentation]]] = None,
) -> List[PredictionRequest]:
    """Plan PM prompts and preserve the exact per-test-input sample budget."""
    variants = _variants_for_protocol(
        task_ids=task_ids,
        protocol=protocol,
        guidance_budget=guidance_budget,
        augmentation_plan=augmentation_plan,
        variants_per_transform=config.variants_per_transform,
    )
    sample_count = _sample_count(
        config=config,
        protocol=protocol,
        guidance_budget=guidance_budget,
    )
    requests: List[PredictionRequest] = []
    test_counts: Dict[str, int] = {}
    for variant in variants:
        count = test_counts.get(variant.task_id)
        if count is None:
            count = len(load_task(variant.task_id, tasks_directory)["test"])
            if count == 0:
                raise ValueError(f"task {variant.task_id!r} has no test inputs.")
            test_counts[variant.task_id] = count
        for test_index in range(count):
            requests.append(
                PredictionRequest(
                    request_id=f"{variant.key}__{test_index}",
                    variant=variant,
                    test_index=test_index,
                    sample_count=sample_count,
                )
            )
    return requests


def candidate_count_by_test_input(
    requests: Sequence[PredictionRequest],
) -> Dict[Tuple[str, int], int]:
    """Return the total number of candidates planned per original test input."""
    totals: Dict[Tuple[str, int], int] = {}
    for request in requests:
        key = (request.variant.task_id, request.test_index)
        totals[key] = totals.get(key, 0) + request.sample_count
    return totals
