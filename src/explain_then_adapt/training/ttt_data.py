"""Deterministic ARC augmentation and exact-span tokenization for online TTT."""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch

from explain_then_adapt.arc.augmented_keys import (
    make_augmented_key,
    parse_augmented_key,
    parse_order_mapping,
)
from explain_then_adapt.arc.formatting import format_grid_to_string
from explain_then_adapt.arc.io import load_task
from explain_then_adapt.arc.transforms import transform_pairs
from explain_then_adapt.arc.types import TRANSFORM_CODES, Example

from .config import TTT_GUIDANCE_BUDGETS, TTTTrainingConfig
from .prediction_data import extract_prediction_guidance

TTT_PLAN_SCHEMA_VERSION = 1
TTT_PLAN_KIND = "ttt_augmentation_plan"


def _stable_rng(seed: int, *parts: str) -> random.Random:
    material = ":".join((str(seed), *parts)).encode("utf-8")
    rng_seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
    return random.Random(rng_seed)


def _normalize_task_id(value: str) -> str:
    task_id = value.strip()
    if task_id.endswith(".json"):
        task_id = task_id[:-5]
    if not task_id or "_" in task_id:
        raise ValueError(
            f"invalid ARC task identifier {value!r}; task IDs must not contain '_'."
        )
    return task_id


def normalize_ttt_task_ids(values: Sequence[str]) -> List[str]:
    """Normalize a non-empty task selection and reject duplicate IDs."""
    task_ids: List[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise ValueError(f"TTT task selection item {index} must be a string.")
        task_ids.append(_normalize_task_id(value))
    if not task_ids:
        raise ValueError("TTT task selection must not be empty.")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("TTT task selection contains duplicate task IDs.")
    return task_ids


def load_ttt_task_ids(path: Path) -> List[str]:
    """Load task IDs from a JSON list or a JSONL task manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"TTT task selection does not exist: {path}.")

    values: List[Any]
    if path.suffix == ".jsonl":
        values = []
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, Mapping):
                    value = value.get("task_id")
                if not isinstance(value, str):
                    raise ValueError(f"{path}:{line_number} has no string task_id.")
                values.append(value)
    else:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, list):
            raise ValueError("TTT task selection JSON must contain a list.")
        values = payload

    return normalize_ttt_task_ids(values)


@dataclass(frozen=True)
class TTTAugmentation:
    """One value/order/geometric augmentation used for a TTT update."""

    task_id: str
    transformation_code: str
    value_mapping: str
    order_mapping: str
    variant_index: int

    def __post_init__(self) -> None:
        normalized_task_id = _normalize_task_id(self.task_id)
        if normalized_task_id != self.task_id:
            raise ValueError("TTT augmentation task_id must already be normalized.")
        if isinstance(self.variant_index, bool) or self.variant_index < 0:
            raise ValueError("TTT augmentation variant_index must be non-negative.")
        if self.transformation_code not in TRANSFORM_CODES:
            raise ValueError(
                f"unsupported TTT transformation {self.transformation_code!r}."
            )
        make_augmented_key(
            self.task_id,
            self.transformation_code,
            self.value_mapping,
            self.order_mapping,
        )

    @property
    def augmented_key(self) -> str:
        return make_augmented_key(
            self.task_id,
            self.transformation_code,
            self.value_mapping,
            self.order_mapping,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "augmented_key": self.augmented_key,
            "task_id": self.task_id,
            "transformation_code": self.transformation_code,
            "value_mapping": self.value_mapping,
            "order_mapping": self.order_mapping,
            "variant_index": self.variant_index,
        }


def _order_mapping(indices: Sequence[int]) -> str:
    if len(indices) <= 10:
        return "".join(str(index) for index in indices)
    return ",".join(str(index) for index in indices)


def _random_value_mapping(rng: random.Random) -> str:
    values = list("0123456789")
    rng.shuffle(values)
    return "".join(values)


def _random_order_mapping(pair_count: int, rng: random.Random) -> str:
    indices = list(range(pair_count))
    rng.shuffle(indices)
    return _order_mapping(indices)


def generate_ttt_augmentation_plan(
    *,
    task_ids: Sequence[str],
    tasks_directory: Path,
    seed: int,
    variants_per_transform: int = 8,
) -> Dict[str, List[TTTAugmentation]]:
    """Generate a stable plan independent of task input order and run name."""
    if variants_per_transform != 8:
        raise ValueError("the final TTT plan requires eight variants per transform.")
    normalized_ids = sorted(normalize_ttt_task_ids(task_ids))
    if not tasks_directory.is_dir():
        raise FileNotFoundError(
            f"ARC task directory does not exist: {tasks_directory}."
        )

    plan: Dict[str, List[TTTAugmentation]] = {}
    for task_id in normalized_ids:
        pairs = load_task(task_id, tasks_directory)["train"]
        if len(pairs) < 2:
            raise ValueError(
                f"TTT task {task_id!r} needs at least two training demonstrations."
            )
        variants: List[TTTAugmentation] = []
        for transformation_index, transformation_code in enumerate(TRANSFORM_CODES):
            rng = _stable_rng(seed, "ttt-plan", task_id, transformation_code)
            seen: Set[Tuple[str, str]] = set()
            for local_index in range(variants_per_transform):
                for _ in range(10_000):
                    value_mapping = _random_value_mapping(rng)
                    order_mapping = _random_order_mapping(len(pairs), rng)
                    identity = (value_mapping, order_mapping)
                    if identity not in seen:
                        seen.add(identity)
                        break
                else:
                    raise RuntimeError(
                        f"could not generate unique TTT variants for {task_id}."
                    )
                variants.append(
                    TTTAugmentation(
                        task_id=task_id,
                        transformation_code=transformation_code,
                        value_mapping=value_mapping,
                        order_mapping=order_mapping,
                        variant_index=(
                            transformation_index * variants_per_transform + local_index
                        ),
                    )
                )
        plan[task_id] = variants
    return plan


def _augmentation_from_value(
    task_id: str,
    value: Any,
    fallback_index: int,
) -> TTTAugmentation:
    if isinstance(value, str):
        parsed = parse_augmented_key(value)
        if parsed.transformation_id is None:
            raise ValueError(f"TTT plan key is not augmented: {value!r}.")
        return TTTAugmentation(
            task_id=parsed.original_key,
            transformation_code=parsed.transformation_id,
            value_mapping=str(parsed.value_mapping),
            order_mapping=str(parsed.order_mapping),
            variant_index=fallback_index,
        )
    if not isinstance(value, Mapping):
        raise ValueError("TTT plan variants must be mappings or augmented keys.")
    augmented_key = value.get("augmented_key")
    if isinstance(augmented_key, str):
        parsed = parse_augmented_key(augmented_key)
        if parsed.transformation_id is None:
            raise ValueError(f"TTT plan key is not augmented: {augmented_key!r}.")
        variant = TTTAugmentation(
            task_id=str(value.get("task_id", parsed.original_key)),
            transformation_code=str(
                value.get("transformation_code", parsed.transformation_id)
            ),
            value_mapping=str(value.get("value_mapping", parsed.value_mapping)),
            order_mapping=str(value.get("order_mapping", parsed.order_mapping)),
            variant_index=int(value.get("variant_index", fallback_index)),
        )
        if variant.augmented_key != augmented_key:
            raise ValueError(
                f"TTT plan fields disagree with augmented_key {augmented_key!r}."
            )
        return variant
    return TTTAugmentation(
        task_id=str(value.get("task_id", task_id)),
        transformation_code=str(value["transformation_code"]),
        value_mapping=str(value["value_mapping"]),
        order_mapping=str(value["order_mapping"]),
        variant_index=int(value.get("variant_index", fallback_index)),
    )


def _validate_and_normalize_task_plan(
    *,
    task_id: str,
    values: Sequence[Any],
    pair_count: int,
    variants_per_transform: int,
) -> List[TTTAugmentation]:
    buckets: Dict[str, List[TTTAugmentation]] = {code: [] for code in TRANSFORM_CODES}
    seen_keys: Set[str] = set()
    for index, value in enumerate(values):
        variant = _augmentation_from_value(task_id, value, index)
        if variant.task_id != task_id:
            raise ValueError(
                f"TTT plan task mismatch: expected {task_id!r}, got "
                f"{variant.task_id!r}."
            )
        order_indices = parse_order_mapping(variant.order_mapping)
        if len(order_indices) != pair_count:
            raise ValueError(
                f"TTT plan order mapping for {variant.augmented_key!r} has "
                f"{len(order_indices)} entries; expected {pair_count}."
            )
        if variant.augmented_key in seen_keys:
            raise ValueError(
                f"TTT plan contains duplicate variant {variant.augmented_key!r}."
            )
        seen_keys.add(variant.augmented_key)
        buckets[variant.transformation_code].append(variant)

    normalized: List[TTTAugmentation] = []
    for transformation_index, transformation_code in enumerate(TRANSFORM_CODES):
        variants = buckets[transformation_code]
        if len(variants) != variants_per_transform:
            raise ValueError(
                f"TTT plan for {task_id!r} has {len(variants)} "
                f"{transformation_code} variants; expected {variants_per_transform}."
            )
        for local_index, variant in enumerate(variants):
            normalized.append(
                TTTAugmentation(
                    task_id=variant.task_id,
                    transformation_code=variant.transformation_code,
                    value_mapping=variant.value_mapping,
                    order_mapping=variant.order_mapping,
                    variant_index=(
                        transformation_index * variants_per_transform + local_index
                    ),
                )
            )
    return normalized


def load_ttt_augmentation_plan(
    path: Path,
    *,
    task_ids: Sequence[str],
    tasks_directory: Path,
    variants_per_transform: int = 8,
) -> Dict[str, List[TTTAugmentation]]:
    """Load the structured plan format or the historical task-to-key mapping."""
    if not path.is_file():
        raise FileNotFoundError(f"TTT augmentation plan does not exist: {path}.")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, Mapping):
        raise ValueError("TTT augmentation plan must contain a mapping.")

    if "kind" in payload or "tasks" in payload:
        if payload.get("schema_version") != TTT_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported TTT augmentation plan schema version.")
        if payload.get("kind") != TTT_PLAN_KIND:
            raise ValueError("unsupported TTT augmentation plan kind.")
        if payload.get("variants_per_transform") != variants_per_transform:
            raise ValueError(
                "TTT augmentation plan variants_per_transform does not match "
                "the run."
            )
        raw_tasks = payload.get("tasks")
    else:
        raw_tasks = payload
    if not isinstance(raw_tasks, Mapping):
        raise ValueError("TTT augmentation plan has no task mapping.")

    plan: Dict[str, List[TTTAugmentation]] = {}
    for raw_task_id in sorted(normalize_ttt_task_ids(task_ids)):
        raw_values = raw_tasks.get(raw_task_id)
        if isinstance(raw_values, Mapping):
            raw_values = raw_values.get("variants")
        if not isinstance(raw_values, list):
            raise ValueError(f"TTT plan has no variant list for {raw_task_id!r}.")
        pair_count = len(load_task(raw_task_id, tasks_directory)["train"])
        if pair_count < 2:
            raise ValueError(
                f"TTT task {raw_task_id!r} needs at least two demonstrations."
            )
        plan[raw_task_id] = _validate_and_normalize_task_plan(
            task_id=raw_task_id,
            values=raw_values,
            pair_count=pair_count,
            variants_per_transform=variants_per_transform,
        )
    return plan


def ttt_augmentation_plan_payload(
    plan: Mapping[str, Sequence[TTTAugmentation]],
    *,
    seed: int,
    variants_per_transform: int,
) -> Dict[str, Any]:
    """Return the readable, versioned representation persisted with each run."""
    return {
        "schema_version": TTT_PLAN_SCHEMA_VERSION,
        "kind": TTT_PLAN_KIND,
        "seed": seed,
        "variants_per_transform": variants_per_transform,
        "tasks": {
            task_id: [variant.to_dict() for variant in variants]
            for task_id, variants in sorted(plan.items())
        },
    }


def selected_guidance_variant_indices(
    guidance_budget: int,
    *,
    variants_per_transform: int = 8,
) -> Tuple[int, ...]:
    """Select a balanced nested subset from every transform block."""
    if isinstance(guidance_budget, bool) or guidance_budget not in TTT_GUIDANCE_BUDGETS:
        raise ValueError(f"guidance budget must be one of {TTT_GUIDANCE_BUDGETS}.")
    if variants_per_transform != 8:
        raise ValueError("the final TTT guidance schedule assumes eight variants.")
    per_transform = guidance_budget // len(TRANSFORM_CODES)
    if per_transform == 0:
        return ()
    stride = variants_per_transform // per_transform
    selected: List[int] = []
    for transformation_index in range(len(TRANSFORM_CODES)):
        block_start = transformation_index * variants_per_transform
        selected.extend(
            block_start + local_index
            for local_index in range(stride - 1, variants_per_transform, stride)
        )
    return tuple(selected)


def _guidance_text(value: Any, key: str) -> str:
    if isinstance(value, Mapping):
        for field in ("guidance", "trace", "normalized_output", "raw_output"):
            candidate = value.get(field)
            if isinstance(candidate, str):
                value = candidate
                break
    if not isinstance(value, str):
        raise ValueError(f"guidance entry {key!r} must resolve to a string.")
    if not value.strip():
        return ""
    if "</think>" in value:
        return extract_prediction_guidance(value)
    guidance = value.strip()
    for heading in ("General natural language description:", "General steps:"):
        if heading not in guidance:
            raise ValueError(f"guidance entry {key!r} is missing {heading!r}.")
    return guidance


def load_ttt_guidance(path: Path) -> Dict[str, str]:
    """Load augmentation-specific guidance from inference JSONL or legacy JSON."""
    if not path.is_file():
        raise FileNotFoundError(f"TTT guidance file does not exist: {path}.")

    if path.suffix.lower() == ".jsonl":
        guidance: Dict[str, str] = {}
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"{path}:{line_number} must contain an object.")
                raw_key = value.get("request_id", value.get("key"))
                if not isinstance(raw_key, str) or not raw_key:
                    raise ValueError(
                        f"{path}:{line_number} has no non-empty request_id."
                    )
                if raw_key in guidance:
                    raise ValueError(
                        f"duplicate TTT guidance key {raw_key!r} in {path}."
                    )
                guidance[raw_key] = _guidance_text(value, raw_key)
        if not guidance:
            raise ValueError(f"TTT guidance JSONL contains no records: {path}.")
        for key in guidance:
            parsed = parse_augmented_key(key)
            if parsed.transformation_id is None:
                raise ValueError(f"TTT guidance key is not augmented: {key!r}.")
        return guidance

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, Mapping):
        raise ValueError("TTT guidance JSON must contain a mapping.")
    for wrapper in ("guidance", "guided"):
        wrapped = payload.get(wrapper)
        if isinstance(wrapped, Mapping):
            payload = wrapped
            break

    guidance: Dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, Mapping) and not any(
            field in value
            for field in ("guidance", "trace", "normalized_output", "raw_output")
        ):
            for nested_key, nested_value in value.items():
                nested_key_string = str(nested_key)
                if nested_key_string in guidance:
                    raise ValueError(
                        f"duplicate TTT guidance key {nested_key_string!r}."
                    )
                guidance[nested_key_string] = _guidance_text(
                    nested_value,
                    nested_key_string,
                )
        else:
            key_string = str(key)
            if key_string in guidance:
                raise ValueError(f"duplicate TTT guidance key {key_string!r}.")
            guidance[key_string] = _guidance_text(value, key_string)
    for key in guidance:
        parsed = parse_augmented_key(key)
        if parsed.transformation_id is None:
            raise ValueError(f"TTT guidance key is not augmented: {key!r}.")
    return guidance


@dataclass(frozen=True)
class ResolvedTTTGuidance:
    content: Optional[str]
    status: str


def resolve_ttt_guidance(
    variant: TTTAugmentation,
    *,
    config: TTTTrainingConfig,
    guidance_by_key: Mapping[str, str],
) -> ResolvedTTTGuidance:
    """Resolve guided, budget-empty, missing, and unguided prompt semantics."""
    if config.profile.guidance_mode == "unguided":
        return ResolvedTTTGuidance(content=None, status="unguided")
    selected = set(
        selected_guidance_variant_indices(
            config.profile.guidance_budget,
            variants_per_transform=config.data.variants_per_transform,
        )
    )
    if variant.variant_index not in selected:
        return ResolvedTTTGuidance(
            content=config.data.empty_guidance_content,
            status="budget_empty",
        )

    guidance = guidance_by_key.get(variant.augmented_key, "")
    if guidance.strip():
        return ResolvedTTTGuidance(content=guidance, status="provided")
    if config.data.missing_guidance_policy == "omit_system":
        return ResolvedTTTGuidance(content=None, status="missing")
    raise ValueError(
        f"selected TTT variant {variant.augmented_key!r} has no guidance; "
        "use missing_guidance_policy=omit_system only for historical compatibility."
    )


def _token_list(value: Any, name: str) -> List[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or any(
        not isinstance(token, int) or isinstance(token, bool) for token in value
    ):
        raise TypeError(f"tokenizer {name} must return a flat list of integers.")
    return value


def _encode(tokenizer: Any, text: str) -> List[int]:
    return _token_list(tokenizer.encode(text, add_special_tokens=False), "encode")


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


def _chat_text(
    *,
    pairs: Sequence[Mapping[str, Any]],
    guidance: Optional[str],
    config: TTTTrainingConfig,
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
    has_system_message: bool,
    tokenizer: Any,
    config: TTTTrainingConfig,
) -> List[List[int]]:
    user_header = _encode(tokenizer, config.data.user_header)
    assistant_header = _encode(tokenizer, config.data.assistant_header)
    message_end = _encode(tokenizer, config.data.message_end)
    cursor = 0
    if has_system_message:
        system_header = _encode(tokenizer, config.data.system_header)
        system_start = _find_subsequence(input_ids, system_header, cursor)
        if system_start != 0:
            raise ValueError("tokenized TTT conversation does not start with system.")
        system_end = _find_subsequence(
            input_ids,
            message_end,
            system_start + len(system_header),
        )
        if system_end < 0:
            raise ValueError("tokenized TTT system message has no end marker.")
        cursor = system_end + len(message_end)

    spans: List[List[int]] = []
    for _ in range(pair_count):
        user_start = _find_subsequence(input_ids, user_header, cursor)
        if user_start < 0:
            raise ValueError("tokenized TTT conversation has too few user turns.")
        user_end = _find_subsequence(
            input_ids,
            message_end,
            user_start + len(user_header),
        )
        if user_end < 0:
            raise ValueError("tokenized TTT user message has no end marker.")
        assistant_header_start = _find_subsequence(
            input_ids,
            assistant_header,
            user_end + len(message_end),
        )
        if assistant_header_start < 0:
            raise ValueError("tokenized TTT conversation has too few assistant turns.")
        assistant_start = assistant_header_start + len(assistant_header)
        assistant_end = _find_subsequence(input_ids, message_end, assistant_start)
        if assistant_end <= assistant_start:
            raise ValueError("tokenized TTT assistant turn has an empty grid target.")
        spans.append([assistant_start, assistant_end])
        cursor = assistant_end + len(message_end)
    if cursor != len(input_ids):
        raise ValueError("tokenized TTT conversation has unexpected trailing turns.")
    return spans


def tokenize_ttt_conversation(
    *,
    tokenizer: Any,
    pairs: Sequence[Mapping[str, Any]],
    variant: TTTAugmentation,
    guidance: ResolvedTTTGuidance,
    config: TTTTrainingConfig,
) -> Dict[str, Any]:
    """Serialize demonstrations and preserve exact assistant-grid token spans."""
    if len(pairs) < 2:
        raise ValueError(
            f"TTT variant {variant.augmented_key!r} needs at least two pairs."
        )
    input_ids = _encode(
        tokenizer,
        _chat_text(pairs=pairs, guidance=guidance.content, config=config),
    )
    if not input_ids:
        raise ValueError(
            f"tokenization produced an empty TTT sequence for "
            f"{variant.augmented_key!r}."
        )
    if len(input_ids) > config.data.max_sequence_length:
        raise ValueError(
            f"TTT variant {variant.augmented_key!r} has {len(input_ids)} tokens; "
            f"limit is {config.data.max_sequence_length}."
        )
    spans = _assistant_spans(
        input_ids=input_ids,
        pair_count=len(pairs),
        has_system_message=guidance.content is not None,
        tokenizer=tokenizer,
        config=config,
    )
    target_spans = spans[1:] if config.data.ignore_first_response else spans
    target_token_count = sum(end - start for start, end in target_spans)
    if target_token_count <= 0:
        raise ValueError(f"TTT variant {variant.augmented_key!r} has no targets.")
    return {
        "variant_id": variant.augmented_key,
        "task_id": variant.task_id,
        "variant_index": variant.variant_index,
        "transformation_code": variant.transformation_code,
        "value_mapping": variant.value_mapping,
        "order_mapping": variant.order_mapping,
        "input_ids": torch.tensor(input_ids, dtype=torch.int32),
        "assistant_spans": spans,
        "sequence_length": len(input_ids),
        "target_token_count": target_token_count,
        "guidance_status": guidance.status,
        "has_system_message": guidance.content is not None,
        "n_pairs_train": len(pairs),
    }


def build_ttt_records(
    *,
    config: TTTTrainingConfig,
    tokenizer: Any,
    tasks_directory: Path,
    task_ids: Sequence[str],
    augmentation_plan: Mapping[str, Sequence[TTTAugmentation]],
    guidance_by_key: Mapping[str, str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Build all 64 token records for every selected ARC task."""
    normalized_ids = sorted(normalize_ttt_task_ids(task_ids))
    if set(augmentation_plan) != set(normalized_ids):
        raise ValueError("TTT augmentation plan task IDs do not match the run.")
    records_by_task: Dict[str, List[Dict[str, Any]]] = {}
    for task_id in normalized_ids:
        pairs: List[Example] = load_task(task_id, tasks_directory)["train"]
        variants = list(augmentation_plan[task_id])
        if len(variants) != config.data.variants_per_task:
            raise ValueError(
                f"TTT task {task_id!r} has {len(variants)} planned variants; "
                f"expected {config.data.variants_per_task}."
            )
        records: List[Dict[str, Any]] = []
        for variant in variants:
            transformed_pairs = transform_pairs(
                pairs,
                variant.transformation_code,
                variant.value_mapping,
                variant.order_mapping,
            )
            guidance = resolve_ttt_guidance(
                variant,
                config=config,
                guidance_by_key=guidance_by_key,
            )
            records.append(
                tokenize_ttt_conversation(
                    tokenizer=tokenizer,
                    pairs=transformed_pairs,
                    variant=variant,
                    guidance=guidance,
                    config=config,
                )
            )
        records_by_task[task_id] = records
    return records_by_task
