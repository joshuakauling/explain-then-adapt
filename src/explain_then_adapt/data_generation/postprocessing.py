"""Normalization helpers for model-produced reasoning traces."""

import re
from typing import Tuple


REQUIRED_SECTIONS: Tuple[str, ...] = (
    "<think>",
    "1) INPUT ANALYSIS",
    "2) OUTPUT ANALYSIS",
    "3) TRANSFORMATION ANALYSIS",
    "4) STEPS FOR THE TRANSFORMATION",
    "</think>",
    "General natural language description:",
    "General steps:",
)


def normalize_trace(text: str) -> str:
    """Normalize harmless formatting drift without inventing missing content."""
    if not isinstance(text, str):
        raise TypeError("trace must be a string.")

    normalized = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u00a0", " ")
        .replace("\u2007", " ")
        .replace("\u202f", " ")
    )
    normalized = normalized.strip()
    if normalized.startswith("```") and normalized.endswith("```"):
        lines = normalized.splitlines()
        if len(lines) >= 2:
            lines = lines[1:-1]
            normalized = "\n".join(lines).strip()

    for section in REQUIRED_SECTIONS:
        escaped = re.escape(section)
        normalized = re.sub(
            rf"(?m)^\s*(?:\*\*|__)?\s*{escaped}\s*(?:\*\*|__)?\s*$",
            section,
            normalized,
        )

    normalized = re.sub(r"(?m)^\s*[\u2022\u2013\u2014]\s+", "- ", normalized)
    normalized = re.sub(r"[ \t]+$", "", normalized, flags=re.MULTILINE)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def section_positions(text: str) -> Tuple[int, ...]:
    """Return first positions for required sections, or ``-1`` when absent."""
    return tuple(text.find(section) for section in REQUIRED_SECTIONS)
