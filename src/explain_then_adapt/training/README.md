# Training

Training covers both model roles used by Explain-then-Adapt and the final online
adaptation stage. The Reasoning Model learns to describe ARC transformations;
the Prediction Model learns to predict output grids with or without that
description; Test-Time Training creates a fresh task-specific adapter at
inference time.

## Shared Setup

Both offline models use the final thesis setup:

- `Qwen/Qwen3-4B-Thinking-2507` with 4-bit NF4 double quantization;
- LoRA rank 128, alpha 32, no dropout, and all attention and MLP projections;
- micro-batches of 2 with 8 accumulation steps, for a global batch of 16;
- a maximum sequence length of 8,192 tokens;
- AdamW 8-bit and cosine decay from `1e-4` to `2e-5` with 5% warmup;
- token-weighted raw and augmented validation every 80 optimizer steps;
- periodic adapter checkpoints every 400 optimizer steps.

Install the GPU dependencies and a Flash Attention build compatible with the
local CUDA and PyTorch versions:

```bash
python -m pip install -e '.[training]'
python -m pip install flash-attn --no-build-isolation
```

## Reasoning Model

The Reasoning Model sees only the ARC demonstration pairs in one user turn.
There is no system message and no test pair. Its assistant target starts with
`<think>`, contains the complete four-part analysis, and ends with the compact
rule and general steps. User and padding tokens are excluded from the loss.

The final settings are versioned in
[`configs/training/reasoning_model.yaml`](../../../configs/training/reasoning_model.yaml).
Each of the 624 tasks contributes one distinct rewritten variant per epoch. The
default 100 variants therefore support 100 epochs without repeating a task
variant.

Build the ignored tokenizer cache from a fresh accepted rewrite run:

```bash
python scripts/build_reasoning_training_data.py \
  --config configs/training/reasoning_model.yaml \
  --tasks-dir /path/to/labelled/arc/tasks \
  --rewrite-requests /path/to/run/rewrite.requests.jsonl \
  --rewrite-results /path/to/run/rewrite.results.jsonl \
  --output-cache data/training/reasoning/reasoning_model.pt \
  --output-manifest data/training/reasoning/reasoning_model.manifest.json
```

The builder rejoins requests and results, validates every trace, reconstructs
its augmented demonstrations, applies the 8,192-token limit, and retains the
configured number of variants per task deterministically. The readable
manifest records source checksums, tokenizer identity, counts, and sequence
lengths.

Start training with:

```bash
python scripts/train_reasoning_model.py \
  --config configs/training/reasoning_model.yaml \
  --cache data/training/reasoning/reasoning_model.pt \
  --output-dir runs/reasoning \
  --run-name reasoning-model-final
```

At 39 optimizer steps per epoch, the final run has 3,900 updates. In addition
to the 400-step interval, explicit epoch checkpoints preserve the thesis
candidates at steps 780 and 1,170. The final adapter is saved as
`end_epoch_99`.

## Prediction Model

The Prediction Model uses a manually serialized Qwen chat conversation. This is
intentional: applying the Thinking model's chat template to each assistant turn
can insert reasoning scaffolding, while the PM target must contain only output
grids.

For a guided sample, everything after the trace's final `</think>` is placed in
one system message. This includes the general natural-language description and
the general steps, but none of the chain of thought. Every labelled training and
test pair from the task is then included as a user input-grid turn followed by
an assistant output-grid turn. All pairs are mixed in one deterministic random
order. The cache stores exact token spans for each output grid, so system
messages, user turns, chat markers, and padding never enter the loss.

Five resolved profiles live in
[`configs/training/prediction_model.yaml`](../../../configs/training/prediction_model.yaml):

| Profile | Source | Guidance | First output | Initialization | Epochs | Extra candidates |
| --- | --- | --- | --- | --- | ---: | --- |
| `guided` | synthetic rewrites | yes | masked | Qwen base | 100 | step 2500 |
| `guided_see_first` | synthetic rewrites | yes | included | Qwen base | 100 | epoch 40 / step 1560 |
| `unguided` | synthetic rewrites | no | masked | Qwen base | 100 | epochs 40 and 60 / steps 1560 and 2340 |
| `unguided_rearc` | ReARC | no | masked | Qwen base | 100 | final only |
| `guided_rearc` | synthetic rewrites | yes | masked | merged Unguided-ReARC model | 40 | final only |

The thesis uses `1570` in parts of the checkpoint discussion but `1560` in the
evaluation table and recovered run paths. With 624 tasks and global batch 16,
the arithmetic is unambiguous: epoch 40 ends at step 1,560. The configuration
records the epoch boundary instead of preserving the inconsistent label.

### Synthetic Caches

Prediction data is rebuilt from the same accepted rewrite request/result corpus
as Reasoning Model data. No historical PM token cache is distributed. Build the
guided cache with:

```bash
python scripts/build_prediction_training_data.py \
  --config configs/training/prediction_model.yaml \
  --profile guided \
  --tasks-dir /path/to/labelled/arc/tasks \
  --rewrite-requests /path/to/run/rewrite.requests.jsonl \
  --rewrite-results /path/to/run/rewrite.results.jsonl \
  --output-cache data/training/prediction/synthetic_guided.pt \
  --output-manifest data/training/prediction/synthetic_guided.manifest.json
```

The same cache is valid for `guided`, `guided_see_first`, and `guided_rearc`;
their tokenized conversations are identical and only the training mask,
initialization, or duration changes. Build a second synthetic cache with
`--profile unguided` for the no-guidance ablation.

### ReARC Cache

ReARC is an optional external source and is not copied into this repository.
The source contract points to
[`michaelhodel/re-arc`](https://github.com/michaelhodel/re-arc/tree/e5b7f1d06362a76f9d3b8c25154ff1fafca897ce)
at commit `e5b7f1d06362a76f9d3b8c25154ff1fafca897ce`. Provide the extracted directory
containing 400 JSON task files and 1,000 generated pairs per task:

```bash
python scripts/build_prediction_training_data.py \
  --config configs/training/prediction_model.yaml \
  --profile unguided_rearc \
  --tasks-dir /path/to/labelled/arc/tasks \
  --rearc-tasks-dir /path/to/re_arc/tasks \
  --output-cache data/training/prediction/rearc_unguided.pt \
  --output-manifest data/training/prediction/rearc_unguided.manifest.json
```

The builder verifies the 400-by-1,000 source shape. It packs up to six pairs per
sample under the exact token limit and independently applies a geometric
transformation and value permutation to each pair. Stable SHA-256-derived RNG
seeds replace the process-dependent Python `hash()` used by the legacy script.
The manifest records the expected upstream revision and a digest of the local
JSON directory. The original local ReARC export is unavailable, so byte-level
identity with the historical run cannot be claimed.

### Training Profiles

Train any base-initialized profile with:

```bash
python scripts/train_prediction_model.py \
  --config configs/training/prediction_model.yaml \
  --profile guided \
  --cache data/training/prediction/synthetic_guided.pt \
  --output-dir runs/prediction \
  --run-name prediction-guided
```

Use `--no-wandb` to disable tracking. Existing run directories are never
overwritten.

`guided_rearc` follows the recovered two-stage initialization exactly. First
train `unguided_rearc`, then merge its final adapter into the original Qwen
base:

```bash
python scripts/merge_lora_adapter.py \
  --base-model Qwen/Qwen3-4B-Thinking-2507 \
  --adapter runs/prediction/prediction-unguided-rearc/end_epoch_99 \
  --output-dir models/prediction-unguided-rearc-merged
```

Attach a fresh LoRA adapter to that merged model for the 40-epoch guided stage:

```bash
python scripts/train_prediction_model.py \
  --config configs/training/prediction_model.yaml \
  --profile guided_rearc \
  --cache data/training/prediction/synthetic_guided.pt \
  --initial-model models/prediction-unguided-rearc-merged \
  --output-dir runs/prediction \
  --run-name prediction-guided-rearc
```

## Test-Time Training

Test-Time Training starts from a standalone Prediction Model whose offline LoRA
adapter has already been merged. For each selected ARC task, the runner creates
a fresh rank-32 LoRA adapter and trains it only on augmented versions of that
task's labelled demonstrations. Test inputs and outputs never enter TTT.

The final settings are versioned in
[`configs/training/test_time_training.yaml`](../../../configs/training/test_time_training.yaml):

- eight geometric transforms (`ID`, `R90`, `R180`, `R270`, `FH`, `FV`, `FD1`,
  `FD2`) with eight value/order variants each;
- exactly 64 batch-size-one optimizer updates per task;
- LoRA rank 32, alpha 16, no dropout, no rsLoRA, and all attention and MLP
  projections;
- AdamW 8-bit with gradient clipping at 1.0;
- 50% linear warmup from zero, followed by cosine decay to `5e-6`;
- a guided peak learning rate of `1.25e-4` and an unguided peak of `2e-4`;
- exact assistant-grid spans with the first demonstration output excluded from
  the loss.

Guided TTT places the augmentation-specific rule description and general steps
in a system message before the transformed demonstrations. Budgets `0`, `8`,
`16`, `32`, and `64` select a balanced nested subset from every geometric
transform. Variants outside the selected subset retain the guided prompt shape
with a system message containing exactly one space. This is intentionally
different from the unguided profile, which has no system message at all.

Missing output from the Reasoning Model is also distinct from an unselected
budget item. The default policy fails before GPU training. The
`--missing-guidance-policy omit_system` option reproduces the historical
fallback by omitting the system message only for genuinely missing selected
guidance.

Run guided TTT with:

```bash
python scripts/train_ttt_adapters.py \
  --config configs/training/test_time_training.yaml \
  --profile guided \
  --guidance-budget 32 \
  --tasks /path/to/evaluation_task_ids.json \
  --tasks-dir /path/to/labelled/arc/tasks \
  --prediction-model models/prediction-guided-merged \
  --guidance /path/to/augmentation_guidance.json \
  --output-dir runs/ttt \
  --run-name guided-budget-32
```

`--tasks` accepts either a JSON list or a JSONL manifest with `task_id` fields.
The guidance file can be the JSONL artifact written by
`sample_reasoning_model.py` or a historical JSON mapping from full augmented
keys to compact guidance strings or full traces. Full traces are reduced to the
content after their final `</think>`. Guided budget zero and the `unguided`
profile do not require a guidance file.

By default, augmentation plans are generated deterministically from the config
seed, task ID, and transform. `--augmentation-plan` can instead replay a
structured plan or the historical task-to-augmented-key mapping. Every run saves
the canonical plan, resolved config, tokenizer metadata, aggregate summary, and
source-file digests, plus one atomic adapter directory with a 64-step manifest
per task. `--resume` skips only adapters that have all expected files and an
intact manifest; changed inputs, conflicting settings, or partial output fail
visibly.

This stage creates adapters only. Sampling answer candidates with those adapters
belongs to the separate inference pipeline.

## Training-Loop Fidelity

The rebuilt trainers deliberately preserve the direct PyTorch loop from the
research code: task-balanced epoch selection, gradient accumulation, gradient
clipping, optimizer-before-scheduler updates, W&B metric names, and adapter-only
checkpoints remain explicit. Narrow corrections are documented rather than
hidden:

- validation loss is weighted by assistant-target tokens instead of batch
  means;
- rolling training loss is not reset at epoch boundaries;
- the final partial accumulation group receives a correctly scaled update;
- insufficient variants, malformed records, and cache/config mismatches fail
  before training;
- PM targets use recorded grid spans instead of searching chat headers during
  collation;
- TTT uses the configured cosine floor instead of silently ignoring `eta_min`,
  and serializes complete chat blocks instead of trimming a fixed number of
  characters from each prompt.

## Implementation

- `config.py`: typed RM, PM, and online TTT profiles.
- `qlora.py`: shared quantized model, LoRA, and optimizer construction.
- `reasoning_data.py`: accepted-rewrite join and RM token caches.
- `prediction_data.py`: guided/unguided conversations, ReARC packing, and PM
  cache manifests.
- `reasoning_trainer.py`: Reasoning Model objective and research training loop.
- `prediction_trainer.py`: multi-grid masking and Prediction Model loop.
- `ttt_data.py`: deterministic plans, nested guidance budgets, transformed
  demonstrations, and exact-span TTT records.
- `ttt_trainer.py`: fresh per-task adapters, 64-step scheduling, atomic outputs,
  manifests, and resume handling.
- `model_merge.py`: standalone model creation for the Guided-ReARC handoff.
- `resource_migration.py`: deterministic validation-resource migration.

The CPU suite verifies configuration arithmetic, serialization, masking,
deterministic selection, cache compatibility, TTT budget semantics, scheduling,
validation weighting, checkpoint triggers, atomic task adapters, and resume with
tiny models. Full training still requires CUDA, the optional training stack,
labelled ARC task files, and newly generated rewrite data; no unavailable
historical checkpoint is required or claimed.
