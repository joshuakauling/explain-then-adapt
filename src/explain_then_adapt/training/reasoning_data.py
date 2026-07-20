"""Build tokenized, task-balanced data for Reasoning Model fine-tuning."""

import hashlib
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import torch

from explain_then_adapt.arc.formatting import format_puzzle_to_string
from explain_then_adapt.arc.io import load_puzzle_train
from explain_then_adapt.arc.transforms import transform_pairs
from explain_then_adapt.arc.types import Example
from explain_then_adapt.data_generation.records import (
    AugmentationSpec,
    GenerationStage,
)
from explain_then_adapt.data_generation.validation import validate_trace_format

from .config import ReasoningTrainingConfig

CACHE_SCHEMA_VERSION = 1
TRAINING_RESOURCE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RewriteRequestSource:
    request_id: str
    task_id: str
    augmentation: AugmentationSpec


@dataclass(frozen=True)
class ReasoningVariant:
    variant_id: str
    task_id: str
    trace: str
    transformation_code: Optional[str] = None
    value_mapping: Optional[str] = None
    order_mapping: Optional[str] = None
    source_request_id: Optional[str] = None

    def __post_init__(self) -> None:
        fields = (
            self.transformation_code,
            self.value_mapping,
            self.order_mapping,
        )
        if any(value is None for value in fields) and not all(
            value is None for value in fields
        ):
            raise ValueError(
                "augmentation fields must be either all present or all absent."
            )

    @property
    def is_augmented(self) -> bool:
        return self.transformation_code is not None

    def augmentation_dict(self) -> Optional[Dict[str, str]]:
        if not self.is_augmented:
            return None
        assert self.transformation_code is not None
        assert self.value_mapping is not None
        assert self.order_mapping is not None
        return {
            "transformation_code": self.transformation_code,
            "value_mapping": self.value_mapping,
            "order_mapping": self.order_mapping,
        }


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on line {line_number} of {path}."
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"line {line_number} of {path} must contain an object."
                )
            yield value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_ids(path: Path) -> List[str]:
    task_ids: List[str] = []
    for record in _iter_jsonl(path):
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("every task-manifest record requires a non-empty task_id.")
        task_ids.append(task_id)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task manifest contains duplicate task IDs.")
    if not task_ids:
        raise ValueError("task manifest must not be empty.")
    return task_ids


def _load_rewrite_request_index(
    paths: Sequence[Path],
) -> Dict[str, RewriteRequestSource]:
    index: Dict[str, RewriteRequestSource] = {}
    for path in paths:
        for record in _iter_jsonl(path):
            if record.get("stage") != GenerationStage.REWRITE.value:
                raise ValueError(f"{path} contains a non-rewrite request.")
            request_id = record.get("request_id")
            task_id = record.get("task_id")
            augmentation = record.get("augmentation")
            if not isinstance(request_id, str) or not request_id.strip():
                raise ValueError(f"{path} contains a request without request_id.")
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(f"rewrite request {request_id!r} has no task_id.")
            if not isinstance(augmentation, Mapping):
                raise ValueError(f"rewrite request {request_id!r} has no augmentation.")
            source = RewriteRequestSource(
                request_id=request_id,
                task_id=task_id,
                augmentation=AugmentationSpec.from_dict(augmentation),
            )
            previous = index.get(request_id)
            if previous is not None and previous != source:
                raise ValueError(
                    f"conflicting duplicate rewrite request: {request_id!r}."
                )
            index[request_id] = source
    if not index:
        raise ValueError("no rewrite requests were found.")
    return index


def _static_accepted(record: Mapping[str, Any]) -> bool:
    validation = record.get("validation")
    if not isinstance(validation, Mapping):
        return False
    static = validation.get("static")
    return isinstance(static, Mapping) and static.get("accepted") is True


def _iter_accepted_rewrite_variants(
    paths: Sequence[Path],
    request_index: Mapping[str, RewriteRequestSource],
    stats: Dict[str, int],
) -> Iterator[ReasoningVariant]:
    seen_results: Dict[str, str] = {}
    for path in paths:
        for record in _iter_jsonl(path):
            stats["result_records"] += 1
            if record.get("stage") != GenerationStage.REWRITE.value:
                raise ValueError(f"{path} contains a non-rewrite result.")
            request_id = record.get("request_id")
            if not isinstance(request_id, str) or request_id not in request_index:
                raise ValueError(
                    f"rewrite result has no matching request: {request_id!r}."
                )

            fingerprint = hashlib.sha256(
                json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            previous_fingerprint = seen_results.get(request_id)
            if previous_fingerprint is not None:
                if previous_fingerprint != fingerprint:
                    raise ValueError(
                        f"conflicting duplicate rewrite result: {request_id!r}."
                    )
                stats["duplicate_result_records"] += 1
                continue
            seen_results[request_id] = fingerprint

            if record.get("error") is not None or not _static_accepted(record):
                stats["rejected_results"] += 1
                continue

            source = request_index[request_id]
            task_id = record.get("task_id")
            if task_id != source.task_id:
                raise ValueError(f"task mismatch for rewrite result {request_id!r}.")
            result_augmentation = record.get("augmentation")
            if isinstance(result_augmentation, Mapping):
                if (
                    AugmentationSpec.from_dict(result_augmentation)
                    != source.augmentation
                ):
                    raise ValueError(
                        f"augmentation mismatch for rewrite result {request_id!r}."
                    )

            output = record.get("normalized_output") or record.get("raw_output")
            if not isinstance(output, str) or not output.strip():
                raise ValueError(
                    f"accepted rewrite result {request_id!r} has no output."
                )
            validation = validate_trace_format(output)
            if not validation.accepted:
                raise ValueError(
                    f"accepted rewrite result {request_id!r} fails current validation."
                )
            stats["accepted_results"] += 1
            spec = source.augmentation
            yield ReasoningVariant(
                variant_id=request_id,
                task_id=source.task_id,
                trace=validation.normalized_text,
                transformation_code=spec.transformation_code,
                value_mapping=spec.value_mapping,
                order_mapping=spec.order_mapping,
                source_request_id=request_id,
            )


def _load_validation_variants(
    path: Path, expected_split: str
) -> List[ReasoningVariant]:
    variants: List[ReasoningVariant] = []
    seen_variant_ids: Set[str] = set()
    seen_task_ids: Set[str] = set()
    for record in _iter_jsonl(path):
        if int(record.get("schema_version", -1)) != TRAINING_RESOURCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported validation resource schema in {path}.")
        if record.get("split") != expected_split:
            raise ValueError(f"unexpected validation split in {path}.")
        variant_id = record.get("variant_id")
        task_id = record.get("task_id")
        trace = record.get("trace")
        if not isinstance(variant_id, str) or not variant_id:
            raise ValueError(f"validation record in {path} has no variant_id.")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"validation record {variant_id!r} has no task_id.")
        if variant_id in seen_variant_ids or task_id in seen_task_ids:
            raise ValueError(f"duplicate validation record in {path}: {variant_id!r}.")
        if not isinstance(trace, str) or not validate_trace_format(trace).accepted:
            raise ValueError(f"validation trace {variant_id!r} is malformed.")

        augmentation = record.get("augmentation")
        if augmentation is None:
            variant = ReasoningVariant(variant_id, task_id, trace)
        elif isinstance(augmentation, Mapping):
            variant = ReasoningVariant(
                variant_id=variant_id,
                task_id=task_id,
                trace=trace,
                transformation_code=str(augmentation["transformation_code"]),
                value_mapping=str(augmentation["value_mapping"]),
                order_mapping=str(augmentation["order_mapping"]),
            )
        else:
            raise ValueError(
                f"invalid augmentation in validation record {variant_id!r}."
            )
        seen_variant_ids.add(variant_id)
        seen_task_ids.add(task_id)
        variants.append(variant)
    if not variants:
        raise ValueError(f"validation resource is empty: {path}.")
    return variants


def _materialize_puzzle(
    variant: ReasoningVariant,
    tasks_directory: Path,
    puzzle_cache: Dict[str, List[Example]],
) -> List[Example]:
    puzzle = puzzle_cache.get(variant.task_id)
    if puzzle is None:
        puzzle = load_puzzle_train(variant.task_id, tasks_directory)
        puzzle_cache[variant.task_id] = puzzle
    if not variant.is_augmented:
        return puzzle
    assert variant.transformation_code is not None
    assert variant.value_mapping is not None
    assert variant.order_mapping is not None
    return transform_pairs(
        puzzle,
        variant.transformation_code,
        variant.value_mapping,
        variant.order_mapping,
    )


def _token_list(value: Any, name: str) -> List[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or any(
        not isinstance(token, int) or isinstance(token, bool) for token in value
    ):
        raise TypeError(f"tokenizer {name} must return a flat list of integers.")
    return value


def _find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> int:
    """Return the index immediately after the first matching subsequence."""
    if not subsequence or len(subsequence) > len(sequence):
        return -1
    width = len(subsequence)
    for index in range(len(sequence) - width + 1):
        if sequence[index : index + width] == subsequence:
            return index + width
    return -1


def tokenize_reasoning_variant(
    *,
    tokenizer: Any,
    puzzle: List[Example],
    variant: ReasoningVariant,
    grid_delimiter: str,
    assistant_header: str,
) -> Dict[str, Any]:
    """Tokenize the complete user/assistant conversation like the original run."""
    prompt = format_puzzle_to_string(puzzle, delimiter=grid_delimiter)
    input_ids = _token_list(
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": variant.trace},
            ],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
        ),
        "apply_chat_template",
    )
    assistant_header_ids = _token_list(
        tokenizer.encode(
            assistant_header,
            add_special_tokens=False,
        ),
        "encode",
    )
    assistant_start = _find_subsequence(input_ids, assistant_header_ids)
    if not input_ids:
        raise ValueError(
            f"tokenization produced an empty sequence for {variant.variant_id!r}."
        )
    if assistant_start < 0:
        raise ValueError(
            f"tokenization did not contain the configured assistant header for "
            f"{variant.variant_id!r}."
        )
    if assistant_start >= len(input_ids):
        raise ValueError(
            f"tokenization produced no assistant target for {variant.variant_id!r}."
        )
    record: Dict[str, Any] = {
        "variant_id": variant.variant_id,
        "task_id": variant.task_id,
        "input_ids": torch.tensor(input_ids, dtype=torch.int32),
        "assistant_start": assistant_start,
        "sequence_length": len(input_ids),
    }
    augmentation = variant.augmentation_dict()
    if augmentation is not None:
        record["augmentation"] = augmentation
    if variant.source_request_id is not None:
        record["source_request_id"] = variant.source_request_id
    return record


def _selection_score(seed: int, task_id: str, variant_id: str) -> int:
    value = f"{seed}:{task_id}:{variant_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def load_training_task_ids(path: Path) -> List[str]:
    """Load the ordered task IDs shared by the two offline model builders."""
    return _task_ids(path)


def iter_accepted_rewrite_variants(
    *,
    request_paths: Sequence[Path],
    result_paths: Sequence[Path],
    stats: Dict[str, int],
) -> Iterator[ReasoningVariant]:
    """Join rewrite requests and accepted results into validated variants."""
    return _iter_accepted_rewrite_variants(
        result_paths,
        _load_rewrite_request_index(request_paths),
        stats,
    )


def load_validation_variants(
    path: Path,
    expected_split: str,
) -> List[ReasoningVariant]:
    """Load one versioned Reasoning Model validation view."""
    return _load_validation_variants(path, expected_split)


def variant_selection_score(seed: int, task_id: str, variant_id: str) -> int:
    """Return the stable score used when capping variants per task."""
    return _selection_score(seed, task_id, variant_id)


def _consider_variant(
    heap: List[Tuple[int, str, Dict[str, Any]]],
    record: Dict[str, Any],
    *,
    score: int,
    limit: int,
) -> None:
    item = (-score, str(record["variant_id"]), record)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif score < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _finalize_heaps(
    heaps: Mapping[str, List[Tuple[int, str, Dict[str, Any]]]],
    candidate_counts: Mapping[str, int],
    *,
    variants_per_task: int,
) -> Dict[str, List[Dict[str, Any]]]:
    insufficient = {
        task_id: candidate_counts.get(task_id, 0)
        for task_id in heaps
        if candidate_counts.get(task_id, 0) < variants_per_task
    }
    if insufficient:
        preview = list(sorted(insufficient.items()))[:20]
        raise ValueError(
            f"{len(insufficient)} tasks have fewer than {variants_per_task} valid "
            f"variants after token filtering; first entries: {preview}."
        )
    return {
        task_id: [
            record
            for _, _, record in sorted(
                heap,
                key=lambda item: (-item[0], item[1]),
            )
        ]
        for task_id, heap in sorted(heaps.items())
    }


def _tokenize_validation_split(
    variants: Sequence[ReasoningVariant],
    *,
    tokenizer: Any,
    tasks_directory: Path,
    config: ReasoningTrainingConfig,
    puzzle_cache: Dict[str, List[Example]],
) -> Dict[str, List[Dict[str, Any]]]:
    split: Dict[str, List[Dict[str, Any]]] = {}
    for variant in variants:
        tokenized = tokenize_reasoning_variant(
            tokenizer=tokenizer,
            puzzle=_materialize_puzzle(variant, tasks_directory, puzzle_cache),
            variant=variant,
            grid_delimiter=config.data.grid_delimiter,
            assistant_header=config.data.assistant_header,
        )
        if tokenized["sequence_length"] > config.data.max_sequence_length:
            raise ValueError(
                f"validation variant {variant.variant_id!r} exceeds "
                f"{config.data.max_sequence_length} tokens."
            )
        split[variant.task_id] = [tokenized]
    return dict(sorted(split.items()))


def load_reasoning_tokenizer(config: ReasoningTrainingConfig) -> Any:
    """Load the tokenizer pinned by the Reasoning Model configuration."""
    from transformers import AutoTokenizer

    kwargs: Dict[str, Any] = {}
    if config.model.revision is not None:
        kwargs["revision"] = config.model.revision
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, **kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Reasoning Model tokenizer has neither pad nor EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_reasoning_token_cache(
    *,
    config: ReasoningTrainingConfig,
    tokenizer: Any,
    tasks_directory: Path,
    task_manifest_path: Path,
    rewrite_request_paths: Sequence[Path],
    rewrite_result_paths: Sequence[Path],
    validation_path: Path,
    augmented_validation_path: Path,
    output_cache_path: Path,
    output_manifest_path: Path,
) -> Mapping[str, Any]:
    """Create one external token cache with train, val, and val-aug splits."""
    expected_task_ids = _task_ids(task_manifest_path)
    expected_task_set = set(expected_task_ids)
    request_index = _load_rewrite_request_index(rewrite_request_paths)
    stats = {
        "result_records": 0,
        "duplicate_result_records": 0,
        "rejected_results": 0,
        "accepted_results": 0,
        "dropped_too_long": 0,
    }
    heaps: Dict[str, List[Tuple[int, str, Dict[str, Any]]]] = {
        task_id: [] for task_id in expected_task_ids
    }
    candidate_counts: Dict[str, int] = {task_id: 0 for task_id in expected_task_ids}
    puzzle_cache: Dict[str, List[Example]] = {}

    for variant in _iter_accepted_rewrite_variants(
        rewrite_result_paths,
        request_index,
        stats,
    ):
        if variant.task_id not in expected_task_set:
            raise ValueError(
                f"rewrite result belongs to task outside the manifest: {variant.task_id!r}."
            )
        tokenized = tokenize_reasoning_variant(
            tokenizer=tokenizer,
            puzzle=_materialize_puzzle(variant, tasks_directory, puzzle_cache),
            variant=variant,
            grid_delimiter=config.data.grid_delimiter,
            assistant_header=config.data.assistant_header,
        )
        if tokenized["sequence_length"] > config.data.max_sequence_length:
            stats["dropped_too_long"] += 1
            continue
        candidate_counts[variant.task_id] += 1
        score = _selection_score(config.seed, variant.task_id, variant.variant_id)
        _consider_variant(
            heaps[variant.task_id],
            tokenized,
            score=score,
            limit=config.data.variants_per_task,
        )

    training_split = _finalize_heaps(
        heaps,
        candidate_counts,
        variants_per_task=config.data.variants_per_task,
    )
    validation_variants = _load_validation_variants(validation_path, "validation")
    augmented_validation_variants = _load_validation_variants(
        augmented_validation_path,
        "validation_augmented",
    )
    validation_task_ids = {variant.task_id for variant in validation_variants}
    augmented_validation_task_ids = {
        variant.task_id for variant in augmented_validation_variants
    }
    if validation_task_ids != augmented_validation_task_ids:
        raise ValueError("validation and augmented validation task IDs differ.")
    overlap = sorted(expected_task_set & validation_task_ids)
    if overlap:
        raise ValueError(f"validation tasks overlap the training manifest: {overlap}.")
    validation_split = _tokenize_validation_split(
        validation_variants,
        tokenizer=tokenizer,
        tasks_directory=tasks_directory,
        config=config,
        puzzle_cache=puzzle_cache,
    )
    augmented_validation_split = _tokenize_validation_split(
        augmented_validation_variants,
        tokenizer=tokenizer,
        tasks_directory=tasks_directory,
        config=config,
        puzzle_cache=puzzle_cache,
    )

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if not isinstance(pad_token_id, int):
        raise ValueError(
            "tokenizer.pad_token_id must be set before building the cache."
        )
    tokenizer_kwargs = getattr(tokenizer, "init_kwargs", {})
    tokenizer_revision = (
        tokenizer_kwargs.get("_commit_hash")
        if isinstance(tokenizer_kwargs, Mapping)
        else None
    )
    sources = {
        "task_manifest": _sha256(task_manifest_path),
        "rewrite_requests": {
            str(path): _sha256(path) for path in rewrite_request_paths
        },
        "rewrite_results": {str(path): _sha256(path) for path in rewrite_result_paths},
        "validation": _sha256(validation_path),
        "validation_augmented": _sha256(augmented_validation_path),
    }
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "config": config.to_dict(),
        "tokenizer": {
            "name": str(getattr(tokenizer, "name_or_path", config.model.name)),
            "revision": tokenizer_revision or config.model.revision,
            "pad_token_id": pad_token_id,
        },
        "sources": sources,
    }
    cache = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "metadata": metadata,
        "splits": {
            "train": training_split,
            "validation": validation_split,
            "validation_augmented": augmented_validation_split,
        },
    }
    output_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_cache_path)

    train_lengths = [
        int(record["sequence_length"])
        for records in training_split.values()
        for record in records
    ]
    manifest: Dict[str, Any] = {
        **metadata,
        "cache": {
            "path": str(output_cache_path),
            "sha256": _sha256(output_cache_path),
        },
        "counts": {
            **stats,
            "train_tasks": len(training_split),
            "train_variants": sum(len(records) for records in training_split.values()),
            "validation_tasks": len(validation_split),
            "validation_augmented_tasks": len(augmented_validation_split),
        },
        "train_sequence_lengths": {
            "minimum": min(train_lengths),
            "maximum": max(train_lengths),
            "mean": sum(train_lengths) / len(train_lengths),
        },
        "valid_candidates_per_task": dict(sorted(candidate_counts.items())),
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    return manifest
