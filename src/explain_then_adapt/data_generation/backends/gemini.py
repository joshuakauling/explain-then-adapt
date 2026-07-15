"""Gemini Batch API serialization and optional SDK operations."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from ..records import (
    GenerationRequest,
    GenerationResult,
    TokenUsage,
    read_jsonl,
    write_jsonl,
)


PathLike = Union[str, Path]


def _part(text: str) -> Dict[str, str]:
    return {"text": text}


def to_gemini_batch_record(request: GenerationRequest) -> Dict[str, Any]:
    """Serialize an internal request as one Gemini Batch API JSONL record."""
    system_messages = [
        message.content for message in request.messages if message.role == "system"
    ]
    contents = [
        {
            "role": "model" if message.role == "assistant" else "user",
            "parts": [_part(message.content)],
        }
        for message in request.messages
        if message.role != "system"
    ]
    provider_request: Dict[str, Any] = {"contents": contents}
    if not request.sampling.use_provider_defaults:
        provider_request["generation_config"] = {
            "temperature": request.sampling.temperature,
            "top_p": request.sampling.top_p,
            "max_output_tokens": request.sampling.max_tokens,
        }
    if (
        not request.sampling.use_provider_defaults
        and request.sampling.seed is not None
    ):
        provider_request["generation_config"]["seed"] = request.sampling.seed
    if system_messages:
        provider_request["system_instruction"] = {
            "parts": [_part("\n\n".join(system_messages))]
        }
    return {"key": request.request_id, "request": provider_request}


def write_gemini_batch_input(
    requests: Iterable[GenerationRequest],
    path: PathLike,
) -> int:
    """Write internal requests in Gemini's inline-request JSONL format."""
    return write_jsonl((to_gemini_batch_record(request) for request in requests), path)


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _response_text(response: Mapping[str, Any]) -> str:
    candidates = response.get("candidates", [])
    if not isinstance(candidates, Sequence) or not candidates:
        return ""
    candidate = candidates[0]
    if not isinstance(candidate, Mapping):
        return ""
    content = candidate.get("content", {})
    if not isinstance(content, Mapping):
        return ""
    parts = content.get("parts", [])
    if not isinstance(parts, Sequence):
        return ""
    return "".join(
        str(part.get("text", "")) for part in parts if isinstance(part, Mapping)
    )


def parse_gemini_batch_record(
    record: Mapping[str, Any],
    request: GenerationRequest,
    *,
    model: str,
) -> GenerationResult:
    """Normalize one success or error record from a Gemini batch result."""
    record_key = str(record.get("key", ""))
    if record_key and record_key != request.request_id:
        raise ValueError(
            f"Gemini result key {record_key!r} does not match request "
            f"{request.request_id!r}."
        )
    response = record.get("response", {})
    if not isinstance(response, Mapping):
        response = {}
    error_value = record.get("error")
    if isinstance(error_value, Mapping):
        error = str(
            _first(error_value, "message", "status", default=dict(error_value))
        )
    elif error_value is not None:
        error = str(error_value)
    else:
        error = None

    usage = response.get("usageMetadata", response.get("usage_metadata", {}))
    if not isinstance(usage, Mapping):
        usage = {}
    candidates = response.get("candidates", [])
    finish_reason: Optional[str] = None
    if isinstance(candidates, Sequence) and candidates and isinstance(candidates[0], Mapping):
        candidate = candidates[0]
        reason = _first(candidate, "finishReason", "finish_reason")
        finish_reason = str(reason) if reason is not None else None

    return GenerationResult(
        request_id=request.request_id,
        task_id=request.task_id,
        stage=request.stage,
        backend="gemini",
        model=model,
        prompt_version=request.prompt_version,
        hint_mode=request.hint_mode,
        few_shot_task_ids=request.few_shot_task_ids,
        sampling=request.sampling,
        augmentation=request.augmentation,
        raw_output=_response_text(response),
        finish_reason=finish_reason,
        usage=TokenUsage(
            prompt_tokens=int(
                _first(usage, "promptTokenCount", "prompt_token_count", default=0)
            ),
            completion_tokens=int(
                _first(
                    usage,
                    "candidatesTokenCount",
                    "candidates_token_count",
                    default=0,
                )
            ),
        ),
        error=error,
        metadata={
            **dict(request.metadata),
            "provider_record_key": record_key or request.request_id,
        },
    )


def read_gemini_batch_results(
    path: PathLike,
    requests: Sequence[GenerationRequest],
    *,
    model: str,
) -> List[GenerationResult]:
    """Join Gemini results to internal requests by stable request identifier."""
    by_id = {request.request_id: request for request in requests}
    results: List[GenerationResult] = []
    for record in read_jsonl(path):
        request_id = str(record.get("key", ""))
        try:
            request = by_id[request_id]
        except KeyError as error:
            raise ValueError(f"unknown Gemini result key: {request_id!r}.") from error
        results.append(parse_gemini_batch_record(record, request, model=model))
    return results


def create_client(api_key: Optional[str] = None) -> Any:
    """Create a Gemini SDK client while keeping the dependency optional."""
    try:
        from dotenv import find_dotenv, load_dotenv
        from google import genai
    except ImportError as error:
        raise RuntimeError(
            "Gemini execution requires the optional 'gemini' dependencies. "
            "Install the project with: pip install -e '.[gemini]'"
        ) from error
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    return genai.Client(api_key=api_key) if api_key else genai.Client()


def submit_batch(
    input_path: PathLike,
    *,
    model: str,
    display_name: str,
    api_key: Optional[str] = None,
) -> Any:
    """Upload a JSONL file and submit it to the Gemini Batch API."""
    client = create_client(api_key)
    try:
        uploaded_file = client.files.upload(
            file=str(input_path),
            config={"display_name": display_name, "mime_type": "application/jsonl"},
        )
        return client.batches.create(
            model=model,
            src=uploaded_file.name,
            config={"display_name": display_name},
        )
    finally:
        client.close()


def get_batch(batch_name: str, *, api_key: Optional[str] = None) -> Any:
    """Retrieve the current Gemini batch job state."""
    client = create_client(api_key)
    try:
        return client.batches.get(name=batch_name)
    finally:
        client.close()


def download_batch_results(
    batch_name: str,
    output_path: PathLike,
    *,
    api_key: Optional[str] = None,
) -> Path:
    """Download the completed Gemini batch destination file."""
    client = create_client(api_key)
    try:
        batch = client.batches.get(name=batch_name)
        destination = getattr(batch, "dest", None)
        file_name = getattr(destination, "file_name", None)
        if not file_name:
            raise RuntimeError("Gemini batch has no downloadable destination file.")
        content = client.files.download(file=file_name)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            output.write_bytes(content)
        elif hasattr(content, "read"):
            output.write_bytes(content.read())
        else:
            output.write_bytes(bytes(content))
        return output
    finally:
        client.close()
