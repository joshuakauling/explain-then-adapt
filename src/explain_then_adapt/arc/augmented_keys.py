"""Parsing and construction of ARC augmentation identifiers."""

from typing import List, NamedTuple, Optional, Sequence, TypeVar

from .types import TRANSFORM_CODES


T = TypeVar("T")


class AugmentedKey(NamedTuple):
    """Parsed components of a plain or augmented ARC task identifier."""

    original_key: str
    transformation_id: Optional[str]
    value_mapping: Optional[str]
    order_mapping: Optional[str]


def _validate_value_mapping(value_mapping: str) -> None:
    if len(value_mapping) != 10 or not value_mapping.isascii() or not value_mapping.isdigit():
        raise ValueError("value_mapping must contain exactly 10 ASCII digits.")
    if set(value_mapping) != set("0123456789"):
        raise ValueError("value_mapping must be a permutation of digits 0..9.")


def _validate_order_indices(indices: List[int]) -> None:
    if sorted(indices) != list(range(len(indices))):
        raise ValueError(
            f"order_mapping must be a permutation of 0..{len(indices) - 1}: {indices}"
        )


def parse_order_mapping(order_mapping: str) -> List[int]:
    """Parse compact, comma-separated, or whitespace-separated order indices."""
    normalized = order_mapping.strip()
    if not normalized:
        raise ValueError("order_mapping must not be empty.")

    if "," in normalized:
        parts = [part.strip() for part in normalized.split(",")]
    elif any(character.isspace() for character in normalized):
        parts = normalized.split()
    else:
        parts = list(normalized)

    if not parts or any(not part.isascii() or not part.isdigit() for part in parts):
        raise ValueError(f"unsupported order_mapping format: {order_mapping!r}")

    indices = [int(part) for part in parts]
    _validate_order_indices(indices)
    return indices


def apply_order_mapping(
    pairs: Sequence[T],
    order_mapping: Optional[str],
) -> List[T]:
    """Return pairs in the requested order without modifying the input sequence."""
    if order_mapping is None or not order_mapping.strip():
        return list(pairs)

    indices = parse_order_mapping(order_mapping)
    if len(indices) != len(pairs):
        raise ValueError(
            "order_mapping length mismatch: "
            f"got {len(indices)} indices for {len(pairs)} pairs."
        )
    return [pairs[index] for index in indices]


def parse_augmented_key(key: str) -> AugmentedKey:
    """Parse a plain task key or a four-part augmented task key."""
    parts = key.split("_")
    if len(parts) == 1 and parts[0]:
        return AugmentedKey(parts[0], None, None, None)
    if len(parts) != 4 or any(not part for part in parts):
        raise ValueError(f"cannot parse ARC key: {key!r}")

    original_key, transformation_id, value_mapping, order_mapping = parts
    if transformation_id not in TRANSFORM_CODES:
        raise ValueError(f"unknown transformation code: {transformation_id!r}")
    _validate_value_mapping(value_mapping)
    parse_order_mapping(order_mapping)
    return AugmentedKey(
        original_key,
        transformation_id,
        value_mapping,
        order_mapping,
    )


def is_augmented_key(key: str) -> bool:
    """Return whether a key is a valid augmented ARC task identifier."""
    try:
        parsed = parse_augmented_key(key)
    except (TypeError, ValueError):
        return False
    return parsed.transformation_id is not None


def make_augmented_key(
    original_key: str,
    transformation_id: str,
    value_mapping: str,
    order_mapping: str,
) -> str:
    """Build and validate an augmented ARC task identifier."""
    if not original_key or "_" in original_key:
        raise ValueError("original_key must be non-empty and must not contain underscores.")

    candidate = "_".join(
        (original_key, transformation_id.upper(), value_mapping, order_mapping)
    )
    parse_augmented_key(candidate)
    return candidate
