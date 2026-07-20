# Inference

Inference implements the final candidate-generation protocols from the thesis.
It connects three separately restartable stages:

1. build one deterministic 64-variant augmentation plan;
2. sample one Reasoning Model trace for every task view that needs guidance;
3. generate Prediction Model candidates, optionally through the per-task LoRA
   adapters created by Test-Time Training.

The pipeline deliberately stops at raw candidate generation. Grid parsing,
inverse transformation into the original task space, exact scoring, and budget
aggregation belong to `evaluation/`. The thesis did not use a separate
candidate selector or verifier, so inference does not introduce one.

## Setup

Install the optional vLLM stack on a compatible GPU machine:

```bash
python -m pip install -e '.[vllm]'
```

Both model arguments expect standalone Hugging Face models. Merge the offline
LoRA into its base model before inference. TTT adapters remain separate and are
loaded per original ARC task at Prediction Model inference time.

The final defaults live in
[`configs/inference/inference.yaml`](../../../configs/inference/inference.yaml).
Model paths, ARC paths, adapter paths, and outputs are explicit CLI arguments.

## Protocols

Candidate budgets apply independently to every test input in a task.

| Protocol | Task views | Samples per view | Candidates per test input |
| --- | ---: | ---: | ---: |
| `standard32` | original task | 32 | 32 |
| `augmented64` | all 64 augmentations | 1 | 64 |
| `budgeted64`, `k = 0` | original task | 64 | 64 |
| `budgeted64`, `k > 0` | selected `k` augmentations | `64 / k` | 64 |

Budgets `0`, `8`, `16`, `32`, and `64` use the same nested, transform-balanced
selection as TTT. For example, `k = 8` selects one variant from each of the
eight geometric transform blocks and samples every selected prompt eight times.

The Reasoning Model always produces exactly one trace per required view. New
runs do not reproduce the historical `n=8` RM artifact followed by
`rm_index=0`; they directly generate the single trace that downstream stages
actually consume.

## Shared Plan

Create the plan once:

```bash
python scripts/build_inference_plan.py \
  --config configs/inference/inference.yaml \
  --tasks /path/to/evaluation_task_ids.json \
  --tasks-dir /path/to/arc/tasks \
  --output runs/inference/augmentation_plan.json
```

Pass this exact file to RM sampling, `train_ttt_adapters.py`, and PM sampling.
`standard32` and `budgeted64 --guidance-budget 0` operate on the original task
and therefore do not accept a plan.

## Reasoning Sampling

Sample augmentation-specific guidance for a guided budget-32 run:

```bash
python scripts/sample_reasoning_model.py \
  --config configs/inference/inference.yaml \
  --protocol budgeted64 \
  --guidance-budget 32 \
  --tasks /path/to/evaluation_task_ids.json \
  --tasks-dir /path/to/arc/tasks \
  --augmentation-plan runs/inference/augmentation_plan.json \
  --model /path/to/merged-reasoning-model \
  --output runs/inference/budget32.guidance.jsonl
```

Each JSONL record preserves the full raw trace and extracts only the content
after the final `</think>` as PM guidance. The compact output must contain both
`General natural language description:` and `General steps:`. Malformed traces
remain inspectable in the artifact, but the command fails after writing the run
manifest so they cannot silently reach TTT.

For original-task guidance, use `standard32` or budget zero:

```bash
python scripts/sample_reasoning_model.py \
  --config configs/inference/inference.yaml \
  --protocol standard32 \
  --tasks /path/to/evaluation_task_ids.json \
  --tasks-dir /path/to/arc/tasks \
  --model /path/to/merged-reasoning-model \
  --output runs/inference/standard.guidance.jsonl
```

## Guided TTT

Use the budget-specific augmented guidance to train the adapters:

```bash
python scripts/train_ttt_adapters.py \
  --config configs/training/test_time_training.yaml \
  --profile guided \
  --guidance-budget 32 \
  --tasks /path/to/evaluation_task_ids.json \
  --tasks-dir /path/to/arc/tasks \
  --prediction-model /path/to/merged-prediction-model \
  --guidance runs/inference/budget32.guidance.jsonl \
  --augmentation-plan runs/inference/augmentation_plan.json \
  --output-dir runs/ttt \
  --run-name guided-budget-32
```

Budget zero and unguided TTT do not consume RM guidance. TTT still uses the
shared plan for all 64 updates.

## Prediction Sampling

Run the budgeted candidate protocol through the task adapters:

```bash
python scripts/run_prediction_inference.py \
  --config configs/inference/inference.yaml \
  --protocol budgeted64 \
  --guidance-mode guided \
  --guidance-budget 32 \
  --tasks /path/to/evaluation_task_ids.json \
  --tasks-dir /path/to/arc/tasks \
  --augmentation-plan runs/inference/augmentation_plan.json \
  --guidance runs/inference/budget32.guidance.jsonl \
  --model /path/to/merged-prediction-model \
  --ttt-adapter-root runs/ttt/guided-budget-32 \
  --output runs/inference/budget32.predictions.jsonl
```

For the unguided Aug-64 reference, omit both guidance and adapters as needed:

```bash
python scripts/run_prediction_inference.py \
  --config configs/inference/inference.yaml \
  --protocol augmented64 \
  --guidance-mode unguided \
  --tasks /path/to/evaluation_task_ids.json \
  --tasks-dir /path/to/arc/tasks \
  --augmentation-plan runs/inference/augmentation_plan.json \
  --model /path/to/merged-prediction-model \
  --output runs/inference/unguided-aug64.predictions.jsonl
```

To evaluate adapted parameters on `standard32`, use the same adapter root but
provide `standard.guidance.jsonl` instead of the augmentation-specific guidance.
The two artifacts serve different purposes: augmented guidance trained the TTT
adapter, while original-task guidance conditions final standard sampling.

Every required guidance entry and every requested task adapter is checked before
the PM is loaded. Missing inputs are errors; the rebuilt pipeline does not
silently skip tasks or fall back to the base model.

## Artifacts

RM and PM outputs are streamed to JSONL. A sibling `*.manifest.json` records:

- the resolved configuration and protocol;
- model, task-source, augmentation-plan, and guidance provenance;
- exact prompt and generated token totals;
- request and candidate counts;
- vLLM engine and sampling settings.

PM records retain the raw text, per-sample token counts, test index, original
task ID, full augmentation metadata, prompt hash, and adapter status. Outputs
from augmented requests remain in the augmented coordinate and value space.
Evaluation must parse and invert them before candidates from different variants
are compared.

Completed artifacts are never overwritten. If a process stops before
publication, the dot-prefixed `*.partial` JSONL remains visible for inspection
and is never mistaken for a complete run.

## Implementation

- `config.py`: typed final sampling, engine, serialization, and protocol values;
- `planning.py`: shared plans and exact standard, augmented, and budgeted request
  arithmetic;
- `prompts.py`: RM chat input, raw PM Qwen serialization, and guidance loading;
- `artifacts.py`: streaming JSONL and provenance manifests;
- `vllm_runner.py`: lazy vLLM loading, batched RM inference, and grouped per-task
  LoRA inference.

The CPU tests exercise all protocol budgets, transformed prompt construction,
new and legacy guidance loading, and a complete fake-vLLM budget-8 run. Real
model execution still requires the unavailable checkpoints and GPU hardware.
