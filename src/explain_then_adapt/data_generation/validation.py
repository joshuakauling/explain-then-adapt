"""Validation decisions for generated reasoning traces."""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from .postprocessing import REQUIRED_SECTIONS, normalize_trace, section_positions


JUDGE_VOTE_COUNT = 5
JudgeResponse = Union[str, Mapping[str, Any]]


class JudgeVerdict(str, Enum):
    """Supported verdicts returned by the LLM judge."""

    PASS = "pass"
    FAIL = "fail"


class ValidationRoute(str, Enum):
    """The route through which a trace received its validation decision."""

    JUDGE_5_OF_5 = "judge_5_of_5"
    MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True)
class ValidationResult:
    """A normalized acceptance decision with its provenance."""

    accepted: bool
    route: ValidationRoute
    pass_count: int = 0
    vote_count: int = 0
    reviewer_note: Optional[str] = None

    def to_dict(self) -> Mapping[str, Any]:
        result = {
            "accepted": self.accepted,
            "route": self.route.value,
            "pass_count": self.pass_count,
            "vote_count": self.vote_count,
        }
        if self.reviewer_note is not None:
            result["reviewer_note"] = self.reviewer_note
        return result


@dataclass(frozen=True)
class StaticValidationResult:
    """Schema-only validation result for a normalized reasoning trace."""

    accepted: bool
    normalized_text: str
    missing_sections: Tuple[str, ...]
    sections_in_order: bool
    sections_unique: bool
    single_think_block: bool
    starts_with_think: bool

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "accepted": self.accepted,
            "missing_sections": list(self.missing_sections),
            "sections_in_order": self.sections_in_order,
            "sections_unique": self.sections_unique,
            "single_think_block": self.single_think_block,
            "starts_with_think": self.starts_with_think,
        }


def validate_trace_format(text: str) -> StaticValidationResult:
    """Require all trace tags and headings exactly once and in the expected order."""
    normalized = normalize_trace(text)
    positions = section_positions(normalized)
    missing_sections = tuple(
        section
        for section, position in zip(REQUIRED_SECTIONS, positions)
        if position < 0
    )
    present_positions = [position for position in positions if position >= 0]
    sections_in_order = (
        not missing_sections
        and present_positions == sorted(present_positions)
    )
    sections_unique = all(normalized.count(section) == 1 for section in REQUIRED_SECTIONS)
    single_think_block = (
        normalized.count("<think>") == 1
        and normalized.count("</think>") == 1
    )
    starts_with_think = normalized.startswith("<think>")
    return StaticValidationResult(
        accepted=(
            not missing_sections
            and sections_in_order
            and sections_unique
            and single_think_block
            and starts_with_think
        ),
        normalized_text=normalized,
        missing_sections=missing_sections,
        sections_in_order=sections_in_order,
        sections_unique=sections_unique,
        single_think_block=single_think_block,
        starts_with_think=starts_with_think,
    )


def parse_judge_verdict(response: JudgeResponse) -> JudgeVerdict:
    """Parse a strict ``{"verdict": "pass" | "fail"}`` judge response."""
    if isinstance(response, str):
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as error:
            raise ValueError("judge response must be valid JSON.") from error
    elif isinstance(response, Mapping):
        payload = response
    else:
        raise TypeError("judge response must be a JSON string or mapping.")

    if not isinstance(payload, Mapping):
        raise ValueError("judge response JSON must contain an object.")

    verdict = payload.get("verdict")
    if not isinstance(verdict, str):
        raise ValueError("judge response must contain a string-valued 'verdict'.")

    try:
        return JudgeVerdict(verdict.strip().lower())
    except ValueError as error:
        raise ValueError("judge verdict must be either 'pass' or 'fail'.") from error


def evaluate_judge_responses(
    responses: Sequence[JudgeResponse],
) -> ValidationResult:
    """Accept a trace only when all five independent judge responses pass."""
    if isinstance(responses, (str, bytes)):
        raise TypeError("responses must be a sequence of five judge responses.")
    if len(responses) != JUDGE_VOTE_COUNT:
        raise ValueError(
            f"expected exactly {JUDGE_VOTE_COUNT} judge responses, got {len(responses)}."
        )

    verdicts = [parse_judge_verdict(response) for response in responses]
    pass_count = sum(verdict is JudgeVerdict.PASS for verdict in verdicts)
    return ValidationResult(
        accepted=pass_count == JUDGE_VOTE_COUNT,
        route=ValidationRoute.JUDGE_5_OF_5,
        pass_count=pass_count,
        vote_count=JUDGE_VOTE_COUNT,
    )


def record_manual_review(
    *,
    accepted: bool,
    reviewer_note: str,
) -> ValidationResult:
    """Record an explicit manual decision without presenting it as judge-validated."""
    normalized_note = reviewer_note.strip()
    if not normalized_note:
        raise ValueError("manual review requires a non-empty reviewer note.")
    return ValidationResult(
        accepted=accepted,
        route=ValidationRoute.MANUAL_REVIEW,
        reviewer_note=normalized_note,
    )
