"""Build tokenized, task-balanced data for Prediction Model fine-tuning."""

import hashlib
import heapq
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, cast

import torch

from explain_then_adapt.arc.formatting import format_grid_to_string
from explain_then_adapt.arc.io import load_task
from explain_then_adapt.arc.transforms import transform_example
from explain_then_adapt.arc.types import TRANSFORM_CODES, Example

from .config import PredictionTrainingConfig
from .reasoning_data import (
    ReasoningVariant,
    iter_accepted_rewrite_variants,
    load_training_task_ids,
    load_validation_variants,
    variant_selection_score,
)

PREDICTION_CACHE_SCHEMA_VERSION = 1
REARC_REPOSITORY = "https://github.com/michaelhodel/re-arc"
REARC_REVISION = "e5b7f1d06362a76f9d3b8c25154ff1fafca897ce"
_IDENTITY_VALUE_MAPPING = "0123456789"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(directory: Path) -> str:
    """Hash JSON filenames and contents without depending on the local path."""
    digest = hashlib.sha256()
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError(f"no JSON task files found in {directory}.")
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _token_list(value: Any, name: str) -> List[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or any(
        not isinstance(token, int) or isinstance(token, bool) for token in value
    ):
        raise TypeError(f"tokenizer {name} must return a flat list of integers.")
    return value


def _encode(tokenizer: Any, text: str) -> List[int]:
    return _token_list(
        tokenizer.encode(text, add_special_tokens=False),
        "encode",
    )


def _find_subsequence(
    sequence: Sequence[int],
    subsequence: Sequence[int],
    start: int,
) -> int:
    if not subsequence or start < 0 or len(subsequence) > len(sequence):
        return -1
    width = len(subsequence)
    for index in range(start, len(sequence) - width + 1):
        if sequence[index : index + width] == subsequence:
            return index
    return -1


def _stable_rng(seed: int, *parts: str) -> random.Random:
    material = ":".join((str(seed), *parts)).encode("utf-8")
    rng_seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
    return random.Random(rng_seed)


def extract_prediction_guidance(trace: str) -> str:
    """Return only the rule description and general steps after ``</think>``."""
    _, marker, guidance = trace.rpartition("</think>")
    guidance = guidance.strip()
    if not marker or not guidance:
        raise ValueError("reasoning trace has no guidance after </think>.")
    for heading in ("General natural language description:", "General steps:"):
        if heading not in guidance:
            raise ValueError(f"prediction guidance is missing {heading!r}.")
    return guidance


def _chat_text(
    *,
    pairs: Sequence[Mapping[str, Any]],
    guidance: Optional[str],
    config: PredictionTrainingConfig,
) -> str:
    chunks: List[str] = []
    if guidance is not None:
        chunks.append(f"{config.data.system_header}{guidance}{config.data.message_end}")
    for pair in pairs:
        input_grid = format_grid_to_string(
            pair["input"],
            delimiter=config.data.grid_delimiter,
        )
        output_grid = format_grid_to_string(
            pair["output"],
            delimiter=config.data.grid_delimiter,
        )
        chunks.append(f"{config.data.user_header}{input_grid}{config.data.message_end}")
        chunks.append(
            f"{config.data.assistant_header}{output_grid}{config.data.message_end}"
        )
    return "".join(chunks)


def _assistant_spans(
    *,
    input_ids: Sequence[int],
    pair_count: int,
    guided: bool,
    tokenizer: Any,
    config: PredictionTrainingConfig,
) -> List[List[int]]:
    user_header = _encode(tokenizer, config.data.user_header)
    assistant_header = _encode(tokenizer, config.data.assistant_header)
    message_end = _encode(tokenizer, config.data.message_end)
    cursor = 0
    if guided:
        system_header = _encode(tokenizer, config.data.system_header)
        system_start = _find_subsequence(input_ids, system_header, cursor)
        if system_start != 0:
            raise ValueError(
                "tokenized guided conversation does not start with system."
            )
        system_end = _find_subsequence(
            input_ids,
            message_end,
            system_start + len(system_header),
        )
        if system_end < 0:
            raise ValueError("tokenized system message has no end marker.")
        cursor = system_end + len(message_end)

    spans: List[List[int]] = []
    for _ in range(pair_count):
        user_start = _find_subsequence(input_ids, user_header, cursor)
        if user_start < 0:
            raise ValueError("tokenized conversation has fewer user turns than pairs.")
        user_end = _find_subsequence(
            input_ids,
            message_end,
            user_start + len(user_header),
        )
        if user_end < 0:
            raise ValueError("tokenized user message has no end marker.")
        assistant_header_start = _find_subsequence(
            input_ids,
            assistant_header,
            user_end + len(message_end),
        )
        if assistant_header_start < 0:
            raise ValueError(
                "tokenized conversation has fewer assistant turns than pairs."
            )
        assistant_start = assistant_header_start + len(assistant_header)
        assistant_end = _find_subsequence(input_ids, message_end, assistant_start)
        if assistant_end <= assistant_start:
            raise ValueError("tokenized assistant turn has an empty grid target.")
        spans.append([assistant_start, assistant_end])
        cursor = assistant_end + len(message_end)

    if cursor != len(input_ids):
        raise ValueError("tokenized conversation contains unexpected trailing turns.")
    return spans


def tokenize_prediction_conversation(
    *,
    tokenizer: Any,
    pairs: Sequence[Mapping[str, Any]],
    guidance: Optional[str],
    task_id: str,
    variant_id: str,
    config: PredictionTrainingConfig,
) -> Dict[str, Any]:
    """Serialize Qwen chat blocks and record exact output-grid token spans."""
    if not pairs:
        raise ValueError(f"Prediction Model variant {variant_id!r} has no pairs.")
    guided = guidance is not None
    if guided != config.profile.guided:
        raise ValueError("conversation guidance does not match the selected profile.")
    input_ids = _encode(
        tokenizer,
        _chat_text(pairs=pairs, guidance=guidance, config=config),
    )
    if not input_ids:
        raise ValueError(f"tokenization produced an empty sequence for {variant_id!r}.")
    spans = _assistant_spans(
        input_ids=input_ids,
        pair_count=len(pairs),
        guided=guided,
        tokenizer=tokenizer,
        config=config,
    )
    return {
        "variant_id": variant_id,
        "task_id": task_id,
        "input_ids": torch.tensor(input_ids, dtype=torch.int32),
        "assistant_spans": spans,
        "sequence_length": len(input_ids),
        "guided": guided,
        "n_pairs_total": len(pairs),
        "n_pairs_train": sum(pair.get("split") == "train" for pair in pairs),
        "n_pairs_test": sum(pair.get("split") == "test" for pair in pairs),
    }


def _load_labelled_pairs(task_id: str, tasks_directory: Path) -> List[Dict[str, Any]]:
    task = load_task(task_id, tasks_directory)
    pairs: List[Dict[str, Any]] = []
    for split in ("train", "test"):
        for index, value in enumerate(task[split]):
            if "output" not in value:
                raise ValueError(
                    f"task {task_id!r} {split} pair {index} has no labelled output."
                )
            validated = transform_example(
                cast(Example, value),
                "ID",
                _IDENTITY_VALUE_MAPPING,
            )
            pairs.append({**validated, "split": split})
    if not pairs:
        raise ValueError(f"task {task_id!r} has no labelled pairs.")
    return pairs


def _materialize_synthetic_pairs(
    variant: ReasoningVariant,
    *,
    tasks_directory: Path,
    pair_cache: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    source = pair_cache.get(variant.task_id)
    if source is None:
        source = _load_labelled_pairs(variant.task_id, tasks_directory)
        pair_cache[variant.task_id] = source
    if not variant.is_augmented:
        return [
            {
                "input": [row[:] for row in pair["input"]],
                "output": [row[:] for row in pair["output"]],
                "split": pair["split"],
            }
            for pair in source
        ]

    assert variant.transformation_code is not None
    assert variant.value_mapping is not None
    transformed: List[Dict[str, Any]] = []
    for pair in source:
        value = transform_example(
            cast(Example, pair),
            variant.transformation_code,
            variant.value_mapping,
        )
        transformed.append({**value, "split": pair["split"]})
    return transformed


def _shuffle_pairs(
    pairs: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    variant_id: str,
) -> List[Mapping[str, Any]]:
    shuffled = list(pairs)
    _stable_rng(seed, "pair-order", variant_id).shuffle(shuffled)
    return shuffled


def _tokenize_synthetic_variant(
    variant: ReasoningVariant,
    *,
    tokenizer: Any,
    tasks_directory: Path,
    pair_cache: Dict[str, List[Dict[str, Any]]],
    config: PredictionTrainingConfig,
) -> Dict[str, Any]:
    pairs = _materialize_synthetic_pairs(
        variant,
        tasks_directory=tasks_directory,
        pair_cache=pair_cache,
    )
    record = tokenize_prediction_conversation(
        tokenizer=tokenizer,
        pairs=_shuffle_pairs(
            pairs,
            seed=config.seed,
            variant_id=variant.variant_id,
        ),
        guidance=(
            extract_prediction_guidance(variant.trace)
            if config.profile.guided
            else None
        ),
        task_id=variant.task_id,
        variant_id=variant.variant_id,
        config=config,
    )
    augmentation = variant.augmentation_dict()
    if augmentation is not None:
        record["augmentation"] = augmentation
    if variant.source_request_id is not None:
        record["source_request_id"] = variant.source_request_id
    return record


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
            f"Prediction Model variants; first entries: {preview}."
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


def _build_synthetic_training_split(
    *,
    config: PredictionTrainingConfig,
    tokenizer: Any,
    tasks_directory: Path,
    task_manifest_path: Path,
    rewrite_request_paths: Sequence[Path],
    rewrite_result_paths: Sequence[Path],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int], Dict[str, int]]:
    task_ids = load_training_task_ids(task_manifest_path)
    if len(task_ids) != config.data.synthetic_task_count:
        raise ValueError(
            f"task manifest contains {len(task_ids)} tasks; expected "
            f"{config.data.synthetic_task_count}."
        )
    task_id_set = set(task_ids)
    heaps: Dict[str, List[Tuple[int, str, Dict[str, Any]]]] = {
        task_id: [] for task_id in task_ids
    }
    candidate_counts = {task_id: 0 for task_id in task_ids}
    pair_cache: Dict[str, List[Dict[str, Any]]] = {}
    stats = {
        "result_records": 0,
        "duplicate_result_records": 0,
        "rejected_results": 0,
        "accepted_results": 0,
        "dropped_too_long": 0,
    }
    for variant in iter_accepted_rewrite_variants(
        request_paths=rewrite_request_paths,
        result_paths=rewrite_result_paths,
        stats=stats,
    ):
        if variant.task_id not in task_id_set:
            raise ValueError(
                f"rewrite result belongs to task outside the manifest: "
                f"{variant.task_id!r}."
            )
        record = _tokenize_synthetic_variant(
            variant,
            tokenizer=tokenizer,
            tasks_directory=tasks_directory,
            pair_cache=pair_cache,
            config=config,
        )
        if record["sequence_length"] > config.data.max_sequence_length:
            stats["dropped_too_long"] += 1
            continue
        candidate_counts[variant.task_id] += 1
        _consider_variant(
            heaps[variant.task_id],
            record,
            score=variant_selection_score(
                config.seed,
                variant.task_id,
                variant.variant_id,
            ),
            limit=config.data.variants_per_task,
        )
    return (
        _finalize_heaps(
            heaps,
            candidate_counts,
            variants_per_task=config.data.variants_per_task,
        ),
        stats,
        candidate_counts,
    )


def _random_value_mapping(rng: random.Random) -> str:
    values = list(_IDENTITY_VALUE_MAPPING)
    rng.shuffle(values)
    return "".join(values)


def _load_rearc_pairs(path: Path) -> List[Example]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, list) or not value:
        raise ValueError(f"ReARC task {path} must contain a non-empty pair list.")
    pairs: List[Example] = []
    for index, pair in enumerate(value):
        if not isinstance(pair, dict) or "input" not in pair or "output" not in pair:
            raise ValueError(f"invalid ReARC pair {index} in {path}.")
        pairs.append(
            transform_example(
                cast(Example, pair),
                "ID",
                _IDENTITY_VALUE_MAPPING,
            )
        )
    return pairs


def _build_rearc_training_split(
    *,
    config: PredictionTrainingConfig,
    tokenizer: Any,
    rearc_tasks_directory: Path,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    task_files = sorted(rearc_tasks_directory.glob("*.json"))
    if len(task_files) != config.data.rearc_task_count:
        raise ValueError(
            f"ReARC directory contains {len(task_files)} task files; expected "
            f"{config.data.rearc_task_count}."
        )
    split: Dict[str, List[Dict[str, Any]]] = {}
    stats = {
        "source_pairs": 0,
        "packed_pairs": 0,
        "skipped_oversized_pairs": 0,
    }
    for path in task_files:
        task_id = path.stem
        source_pairs = _load_rearc_pairs(path)
        if len(source_pairs) != config.data.rearc_examples_per_task:
            raise ValueError(
                f"ReARC task {task_id!r} contains {len(source_pairs)} pairs; "
                f"expected {config.data.rearc_examples_per_task}."
            )
        stats["source_pairs"] += len(source_pairs)
        rng = _stable_rng(config.seed, "rearc-augmentation", task_id)
        cursor = 0
        records: List[Dict[str, Any]] = []
        for variant_index in range(config.data.variants_per_task):
            variant_id = f"rearc:{task_id}:{variant_index:03d}"
            packed_pairs: List[Dict[str, Any]] = []
            pair_augmentations: List[Dict[str, Any]] = []
            record: Optional[Dict[str, Any]] = None
            rejected_in_row = 0
            while len(packed_pairs) < config.data.rearc_pairs_per_variant:
                source_index = cursor % len(source_pairs)
                cursor += 1
                transformation_code = rng.choice(TRANSFORM_CODES)
                value_mapping = _random_value_mapping(rng)
                transformed = transform_example(
                    source_pairs[source_index],
                    transformation_code,
                    value_mapping,
                )
                candidate_pairs = [
                    *packed_pairs,
                    {**transformed, "split": "train"},
                ]
                candidate = tokenize_prediction_conversation(
                    tokenizer=tokenizer,
                    pairs=candidate_pairs,
                    guidance=None,
                    task_id=task_id,
                    variant_id=variant_id,
                    config=config,
                )
                if candidate["sequence_length"] > config.data.max_sequence_length:
                    if packed_pairs:
                        break
                    stats["skipped_oversized_pairs"] += 1
                    rejected_in_row += 1
                    if rejected_in_row >= len(source_pairs):
                        raise ValueError(
                            f"no ReARC pair for task {task_id!r} fits the token limit."
                        )
                    continue

                rejected_in_row = 0
                packed_pairs = candidate_pairs
                pair_augmentations.append(
                    {
                        "source_index": source_index,
                        "transformation_code": transformation_code,
                        "value_mapping": value_mapping,
                    }
                )
                record = candidate

            if record is None:
                raise ValueError(f"failed to build ReARC variant {variant_id!r}.")
            record["pair_augmentations"] = pair_augmentations
            records.append(record)
            stats["packed_pairs"] += len(packed_pairs)
        _stable_rng(config.seed, "rearc-variant-order", task_id).shuffle(records)
        split[task_id] = records
    return dict(sorted(split.items())), stats


def _build_validation_split(
    variants: Sequence[ReasoningVariant],
    *,
    tokenizer: Any,
    tasks_directory: Path,
    config: PredictionTrainingConfig,
    pair_cache: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    split: Dict[str, List[Dict[str, Any]]] = {}
    for variant in variants:
        record = _tokenize_synthetic_variant(
            variant,
            tokenizer=tokenizer,
            tasks_directory=tasks_directory,
            pair_cache=pair_cache,
            config=config,
        )
        if record["sequence_length"] > config.data.max_sequence_length:
            raise ValueError(
                f"validation variant {variant.variant_id!r} exceeds "
                f"{config.data.max_sequence_length} tokens."
            )
        split[variant.task_id] = [record]
    return dict(sorted(split.items()))


def prediction_cache_contract(config: PredictionTrainingConfig) -> Dict[str, Any]:
    """Return the settings that change a tokenized Prediction Model cache."""
    return {
        "seed": config.seed,
        "data": {
            "variants_per_task": config.data.variants_per_task,
            "max_sequence_length": config.data.max_sequence_length,
            "grid_delimiter": config.data.grid_delimiter,
            "system_header": config.data.system_header,
            "user_header": config.data.user_header,
            "assistant_header": config.data.assistant_header,
            "message_end": config.data.message_end,
            "synthetic_task_count": config.data.synthetic_task_count,
            "rearc_task_count": config.data.rearc_task_count,
            "rearc_examples_per_task": config.data.rearc_examples_per_task,
            "rearc_pairs_per_variant": config.data.rearc_pairs_per_variant,
        },
        "model": {
            "name": config.model.name,
            "revision": config.model.revision,
        },
        "data_source": config.profile.data_source,
        "guided": config.profile.guided,
    }


def load_prediction_tokenizer(config: PredictionTrainingConfig) -> Any:
    """Load the tokenizer pinned by the Prediction Model configuration."""
    from transformers import AutoTokenizer

    kwargs: Dict[str, Any] = {}
    if config.model.revision is not None:
        kwargs["revision"] = config.model.revision
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, **kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Prediction Model tokenizer has neither pad nor EOS token."
            )
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_prediction_token_cache(
    *,
    config: PredictionTrainingConfig,
    tokenizer: Any,
    tasks_directory: Path,
    validation_path: Path,
    augmented_validation_path: Path,
    output_cache_path: Path,
    output_manifest_path: Path,
    task_manifest_path: Optional[Path] = None,
    rewrite_request_paths: Sequence[Path] = (),
    rewrite_result_paths: Sequence[Path] = (),
    rearc_tasks_directory: Optional[Path] = None,
) -> Mapping[str, Any]:
    """Build a PM cache from accepted rewrites or an external ReARC export."""
    if config.profile.data_source == "synthetic":
        if (
            task_manifest_path is None
            or not rewrite_request_paths
            or not rewrite_result_paths
        ):
            raise ValueError(
                "synthetic PM data requires a task manifest and rewrite requests/results."
            )
        training_split, source_stats, candidate_counts = (
            _build_synthetic_training_split(
                config=config,
                tokenizer=tokenizer,
                tasks_directory=tasks_directory,
                task_manifest_path=task_manifest_path,
                rewrite_request_paths=rewrite_request_paths,
                rewrite_result_paths=rewrite_result_paths,
            )
        )
        sources: Dict[str, Any] = {
            "task_manifest": _sha256(task_manifest_path),
            "rewrite_requests": {
                str(path): _sha256(path) for path in rewrite_request_paths
            },
            "rewrite_results": {
                str(path): _sha256(path) for path in rewrite_result_paths
            },
        }
    else:
        if rearc_tasks_directory is None:
            raise ValueError("the ReARC PM profile requires rearc_tasks_directory.")
        training_split, source_stats = _build_rearc_training_split(
            config=config,
            tokenizer=tokenizer,
            rearc_tasks_directory=rearc_tasks_directory,
        )
        candidate_counts = {
            task_id: len(records) for task_id, records in training_split.items()
        }
        sources = {
            "rearc": {
                "repository": REARC_REPOSITORY,
                "expected_revision": REARC_REVISION,
                "task_directory_sha256": _directory_sha256(rearc_tasks_directory),
            }
        }

    validation_variants = load_validation_variants(validation_path, "validation")
    augmented_variants = load_validation_variants(
        augmented_validation_path,
        "validation_augmented",
    )
    validation_task_ids = {variant.task_id for variant in validation_variants}
    augmented_task_ids = {variant.task_id for variant in augmented_variants}
    if validation_task_ids != augmented_task_ids:
        raise ValueError("validation and augmented validation task IDs differ.")
    overlap = sorted(set(training_split) & validation_task_ids)
    if overlap:
        raise ValueError(f"validation tasks overlap PM training tasks: {overlap}.")
    pair_cache: Dict[str, List[Dict[str, Any]]] = {}
    validation_split = _build_validation_split(
        validation_variants,
        tokenizer=tokenizer,
        tasks_directory=tasks_directory,
        config=config,
        pair_cache=pair_cache,
    )
    augmented_validation_split = _build_validation_split(
        augmented_variants,
        tokenizer=tokenizer,
        tasks_directory=tasks_directory,
        config=config,
        pair_cache=pair_cache,
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
    sources.update(
        {
            "validation": _sha256(validation_path),
            "validation_augmented": _sha256(augmented_validation_path),
        }
    )
    metadata = {
        "schema_version": PREDICTION_CACHE_SCHEMA_VERSION,
        "config": config.to_dict(),
        "cache_contract": prediction_cache_contract(config),
        "tokenizer": {
            "name": str(getattr(tokenizer, "name_or_path", config.model.name)),
            "revision": tokenizer_revision or config.model.revision,
            "pad_token_id": pad_token_id,
        },
        "serialization": "manual_qwen_chat_blocks",
        "sources": sources,
    }
    cache = {
        "schema_version": PREDICTION_CACHE_SCHEMA_VERSION,
        "kind": "prediction_model",
        "metadata": metadata,
        "splits": {
            "train": training_split,
            "validation": validation_split,
            "validation_augmented": augmented_validation_split,
        },
    }
    output_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_cache_path)

    train_records = [
        record for records in training_split.values() for record in records
    ]
    train_lengths = [int(record["sequence_length"]) for record in train_records]
    manifest: Dict[str, Any] = {
        **metadata,
        "cache": {
            "path": str(output_cache_path),
            "sha256": _sha256(output_cache_path),
        },
        "counts": {
            **source_stats,
            "train_tasks": len(training_split),
            "train_variants": len(train_records),
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
