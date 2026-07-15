"""Execution backends for provider-independent generation requests."""

from .gemini import parse_gemini_batch_record, to_gemini_batch_record
from .vllm import to_vllm_messages


__all__ = [
    "parse_gemini_batch_record",
    "to_gemini_batch_record",
    "to_vllm_messages",
]
