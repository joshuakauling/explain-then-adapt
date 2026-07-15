"""Planning and application of trace-aware ARC augmentations."""

import random
from typing import List, Optional, Sequence, Set, Tuple

from explain_then_adapt.arc.transforms import transform_pairs
from explain_then_adapt.arc.types import TRANSFORM_CODES, Example

from .prompts import STYLE_MODES
from .records import AugmentationSpec


DEFAULT_AUGMENTATIONS_PER_TASK = 100
AugmentationSignature = Tuple[str, str, str]


def augmentation_signature(spec: AugmentationSpec) -> AugmentationSignature:
    """Return the transformation fields that define a unique augmented task."""
    return (
        spec.transformation_code,
        spec.value_mapping,
        spec.order_mapping,
    )


def remaining_augmentation_count(
    accepted_count: int,
    target_count: int = DEFAULT_AUGMENTATIONS_PER_TASK,
) -> int:
    """Return how many additional accepted variants are required."""
    if accepted_count < 0 or target_count < 0:
        raise ValueError("accepted_count and target_count must be non-negative.")
    return max(0, target_count - accepted_count)


def _random_value_mapping(rng: random.Random) -> str:
    digits = list("0123456789")
    rng.shuffle(digits)
    return "".join(digits)


def _random_order_mapping(example_count: int, rng: random.Random) -> str:
    indices = list(range(example_count))
    rng.shuffle(indices)
    separator = "" if example_count <= 10 else ","
    return separator.join(str(index) for index in indices)


def plan_augmentation_specs(
    *,
    source_trace_id: str,
    example_count: int,
    accepted_specs: Sequence[AugmentationSpec] = (),
    attempted_specs: Sequence[AugmentationSpec] = (),
    target_count: int = DEFAULT_AUGMENTATIONS_PER_TASK,
    seed: int = 0,
    styles: Optional[Sequence[str]] = None,
) -> List[AugmentationSpec]:
    """Plan one run containing enough new candidates to reach the target.

    Rejected candidates should be included in ``attempted_specs`` on the next call.
    This prevents a deterministic retry from proposing the same malformed rewrite.
    """
    if not source_trace_id.strip():
        raise ValueError("source_trace_id must not be empty.")
    if example_count <= 0:
        raise ValueError("example_count must be positive.")
    requested = remaining_augmentation_count(len(accepted_specs), target_count)
    if requested == 0:
        return []

    selected_styles = tuple(styles) if styles is not None else tuple(STYLE_MODES)
    if not selected_styles or any(style not in STYLE_MODES for style in selected_styles):
        raise ValueError(f"styles must be selected from {tuple(STYLE_MODES)}.")

    prior_specs = tuple(accepted_specs) + tuple(attempted_specs)
    used: Set[AugmentationSignature] = {
        augmentation_signature(spec) for spec in prior_specs
    }
    next_variant_index = (
        max((spec.variant_index for spec in prior_specs), default=-1) + 1
    )
    rng = random.Random(seed)
    planned: List[AugmentationSpec] = []
    max_draws = max(1000, requested * 100)

    for _ in range(max_draws):
        candidate = AugmentationSpec(
            source_trace_id=source_trace_id,
            transformation_code=rng.choice(TRANSFORM_CODES),
            value_mapping=_random_value_mapping(rng),
            order_mapping=_random_order_mapping(example_count, rng),
            style=rng.choice(selected_styles),
            variant_index=next_variant_index + len(planned),
        )
        signature = augmentation_signature(candidate)
        if signature in used:
            continue
        used.add(signature)
        planned.append(candidate)
        if len(planned) == requested:
            return planned

    raise RuntimeError(
        f"could not draw {requested} unique augmentation specifications after "
        f"{max_draws} attempts."
    )


def apply_augmentation(
    puzzle: List[Example],
    spec: AugmentationSpec,
) -> List[Example]:
    """Apply a structured augmentation consistently to all demonstration pairs."""
    return transform_pairs(
        puzzle,
        spec.transformation_code,
        spec.value_mapping,
        spec.order_mapping,
    )
