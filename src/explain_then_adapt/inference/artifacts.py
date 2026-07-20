"""Streaming JSONL artifacts and reproducibility metadata for inference runs."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence, TypeVar

T = TypeVar("T")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_sources_sha256(tasks_directory: Path, task_ids: Sequence[str]) -> str:
    """Hash selected ARC task names and contents without hashing local paths."""
    digest = hashlib.sha256()
    for task_id in sorted(task_ids):
        path = tasks_directory / f"{task_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"ARC task file does not exist: {path}.")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def manifest_path_for(output_path: Path) -> Path:
    if output_path.suffix.lower() != ".jsonl":
        raise ValueError("inference output path must end in .jsonl.")
    return output_path.with_name(f"{output_path.stem}.manifest.json")


def chunked(values: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    if isinstance(size, bool) or size <= 0:
        raise ValueError("chunk size must be a positive integer.")
    for start in range(0, len(values), size):
        yield values[start : start + size]


class JsonlArtifactWriter:
    """Write to a visible partial file and publish only after completion."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.manifest_path = manifest_path_for(output_path)
        self.partial_path = output_path.with_name(f".{output_path.name}.partial")
        for path in (self.output_path, self.manifest_path, self.partial_path):
            if path.exists():
                raise FileExistsError(
                    f"inference artifact already exists and will not be overwritten: "
                    f"{path}."
                )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.partial_path.open("x", encoding="utf-8")
        self._completed = False
        self.record_count = 0

    def write(self, record: Mapping[str, Any]) -> None:
        if self._completed:
            raise RuntimeError("cannot write after completing a JSONL artifact.")
        json.dump(record, self._file, ensure_ascii=False, sort_keys=True)
        self._file.write("\n")
        self._file.flush()
        self.record_count += 1

    def complete(self, manifest: Mapping[str, Any]) -> None:
        if self._completed:
            raise RuntimeError("JSONL artifact is already complete.")
        self._file.close()
        self.partial_path.replace(self.output_path)
        payload: Dict[str, Any] = dict(manifest)
        payload["output_path"] = str(self.output_path)
        payload["output_sha256"] = sha256_file(self.output_path)
        payload["record_count"] = self.record_count

        temporary = self.manifest_path.with_name(f".{self.manifest_path.name}.tmp")
        if temporary.exists():
            raise FileExistsError(f"temporary manifest already exists: {temporary}.")
        with temporary.open("x", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False, sort_keys=True)
            file.write("\n")
        temporary.replace(self.manifest_path)
        self._completed = True

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "JsonlArtifactWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    """Iterate a JSONL artifact with line-aware validation."""
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} must contain an object.")
            yield value
