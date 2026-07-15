# ARC Utilities

This package contains the small, stable ARC primitives that the rest of the
project builds on. It should stay independent from model training, inference
engines, prompt templates, and experiment-specific run logic.

## Why This Comes First

Data generation, training, inference, and evaluation all need the same basic ARC
operations. Keeping them here avoids copying low-level grid and augmentation
logic across pipeline stages.

## Legacy Sources

This implementation consolidates stable behavior from these legacy files:

- `ARC/utils_data.py`
- `ARC/z_gemini/validation_part/utils_data.py`
- local duplicated helpers in dataset and inference scripts, especially
  `parse_key_maybe_augmented`.

## Modules

### `types.py`

- `Grid`: a 2D list of integer ARC cell values.
- `InputExample`, `Example`, and `Task`: typed train/test task structures.
- `TRANSFORM_CODES`: the eight supported geometric transformations.

### `io.py`

- `load_subset`
- `load_task`
- `load_puzzle_train`
- `load_puzzle_test`
- `load_full_puzzle`
- `load_records`
- `load_existing_list`

All loaders accept `str` and `pathlib.Path` paths. They contain no project-local
path assumptions.

### `formatting.py`

- `format_grid_to_string`
- `format_example_to_string`
- `format_puzzle_to_string`
- `format_examples_to_string`
- `load_puzzle_as_string`

Delimiter support is retained because different pipeline stages use both compact
and whitespace-separated grid strings.

### `transforms.py`

- `rotate_grid`
- `flip_grid`
- `transform_grid`
- `parse_value_mapping`
- `remap_grid_by_value_mapping`
- `transform_individual_grid`
- `transform_example`
- `transform_pairs`
- `transform_puzzle_train`
- `transform_full_puzzle`
- `load_and_transform_full_puzzle`

These functions are deterministic, avoid in-place mutation, and use plain Python
lists instead of requiring NumPy.

### `augmented_keys.py`

Legacy scripts repeatedly define helpers for keys shaped like:

```text
<orig_key>_<transformation>_<value_mapping>_<order_mapping>
```

Example:

```text
cc9053aa_FD2_6384521079_012
```

The parsing lives here instead of being duplicated in dataset, inference, and
evaluation scripts.

- `parse_augmented_key`
- `is_augmented_key`
- `make_augmented_key`
- `parse_order_mapping`
- `apply_order_mapping`

## What Should Stay Out

These do not belong in ARC-core:

- prompt templates,
- reasoning-trace postprocessing,
- LLM-as-a-judge validation,
- tokenizer/chat-template logic,
- PyTorch datasets and collators,
- vLLM/OpenAI/Gemini runners,
- Weights & Biases logging,
- run-specific model paths,
- experiment result parsing beyond generic ARC output comparison.

## Package Layout

```text
src/explain_then_adapt/arc/
  __init__.py
  types.py
  io.py
  formatting.py
  transforms.py
  augmented_keys.py
```

## Verification

The focused standard-library test suite covers JSON loading, text formatting,
rectangular-grid transformations, value mappings, example transformations,
augmented-key round trips, and demonstration reordering:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```
