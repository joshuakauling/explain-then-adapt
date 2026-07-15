# Explain-then-Adapt

Combining reasoning traces with test-time training for ARC-AGI.

## Motivation

There are two ARC-AGI ideas that motivated this project.

First, reasoning models can use their chain of thought as a search process: they
compare the demonstrations, look for the hidden transformation, and then apply
the inferred rule to the test input.

Second, the ARC Prize has shown that test-time training can be very effective.
The [ARC Prize 2024 winning ARChitects solution](https://github.com/da-fr/arc-prize-2024/tree/main)
used a fine-tuned model and adapted it at inference time. In this setup, the
few-shot pattern of a test task is augmented with simple geometric
transformations and value permutations, creating task-specific training data for
the model before it predicts the final answer.

Explain-then-Adapt grew out of the question of how these two ideas could be
combined: can a model first explain the likely transformation, and can that
explanation then guide the model while it adapts to the task?

## Core Idea

The pipeline splits the task into two parts:

- **Explain:** a Reasoning Model looks at the demonstration pairs and uses its
  chain of thought to reason about what the transformation could be. It compares
  inputs and outputs, considers the relevant objects, colors, geometry, and
  relations, and finally returns a general natural-language description of the
  transformation together with step-by-step instructions for applying it.
- **Adapt:** this description is placed as additional guidance before the
  demonstration pairs of the task. The Prediction Model is then adapted with
  test-time training on augmented versions of those pairs, using simple
  geometric transformations and value permutations, before predicting the output
  for the test input.

The explanation is not meant to be an executable program. It is a compact rule
hypothesis. The hypothesis behind Explain-then-Adapt is that this rule-level
guidance gives the Prediction Model a better starting point than pure
output-error adaptation, especially when the available test-time compute is
limited.

![Explain-then-Adapt overview](docs/assets/figures/explain_then_adapt.png)

The project follows three research questions:

1. Does reasoning-guided adaptation improve ARC-AGI solve rates compared with
   unguided adaptation?
2. Is a reasoning-based explanation useful on its own, or does the benefit only
   appear when it is combined with test-time training?
3. How does performance change with the available test-time compute budget, and
   where do diminishing returns appear?

## What the Pipeline Needs

To make that question testable, the project needs a few separate pieces:

- **ARC utilities:** load tasks, format grids, apply geometric transformations,
  remap values, and keep small fixtures for tests. See
  [`src/explain_then_adapt/arc/`](src/explain_then_adapt/arc/README.md).
- **Data generation:** create synthetic reasoning traces, validate them, clean
  them up, and augment both the grids and the corresponding explanations.
  See [`src/explain_then_adapt/data_generation/`](src/explain_then_adapt/data_generation/README.md).
- **Generation resources:** keep the final task manifest, curated hints, and
  few-shot traces small, reviewable, and versioned. See
  [`resources/data_generation/`](resources/data_generation/README.md).
- **Training:** train the Reasoning Model, train the Prediction Model, and create
  per-task LoRA adapters during TTT. See
  [`src/explain_then_adapt/training/`](src/explain_then_adapt/training/README.md).
- **Inference:** sample explanations, run guided or unguided prediction, load TTT
  adapters, and generate answer candidates. See
  [`src/explain_then_adapt/inference/`](src/explain_then_adapt/inference/README.md).
- **Evaluation:** parse predicted grids, score exact solves, compare candidates,
  and account for the reasoning, training, and sampling budget. See
  [`src/explain_then_adapt/evaluation/`](src/explain_then_adapt/evaluation/README.md).
- **Configurations and scripts:** keep experiment settings and command-line entry
  points separate from reusable code. See [`configs/`](configs/README.md) and
  [`scripts/`](scripts/README.md).

Each component keeps its detailed setup notes and run instructions close to the
code it describes.

## Workflow

```text
ARC tasks
  -> reasoning trace generation
  -> reasoning model training
  -> prediction model training
  -> per-task test-time training
  -> inference
  -> evaluation and budget analysis
```

In practice, this means the project has an offline part and an online part.

The offline part builds the data and trains the two model roles:

1. generate and validate reasoning traces,
2. train the Reasoning Model on those traces,
3. train the Prediction Model on ARC-style input-output conversations.

The online part happens per task at inference time:

1. sample an explanation for the current task,
2. create augmented versions of the task's few-shot examples,
3. adapt the Prediction Model with TTT,
4. predict candidates for the test input,
5. score the result and account for the compute budget.

## Repository Layout

```text
configs/                  Versioned experiment and pipeline configurations.
docs/                     Methodology, results, migration notes, and figures.
notebooks/                Exploratory notebooks only.
resources/                Small versioned manifests and curated prompt inputs.
scripts/                  Command-line entry points for the main workflows.
src/explain_then_adapt/   Reusable Python package.
tests/                    Unit tests, smoke tests, and small fixtures.
```

The package itself is organized around the project pipeline:

```text
src/explain_then_adapt/
  arc/              ARC task loading, formatting, transformations, and types.
  data_generation/  Reasoning trace generation, augmentation, and validation.
  training/         RM, PM, and test-time adapter training.
  inference/        Reasoning sampling, prediction sampling, TTT inference.
  evaluation/       Prediction parsing, scoring, solve rate, budget metrics.
```

## Local Setup

The reusable ARC core has no third-party runtime dependencies. Install the
package in editable mode from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Run the current test suite with:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Provider-specific data-generation dependencies are optional:

```bash
python -m pip install -e '.[gemini]'
python -m pip install -e '.[vllm]'
```

This keeps the ARC core and its tests independent of API clients and the local
GPU stack.

For Gemini access, create a local `.env` from `.env.example` and set:

```dotenv
GEMINI_API_KEY=your_key_here
```

The Gemini CLI loads this file automatically. `.env` is ignored by Git; API
keys should not be passed as command-line arguments or committed to the repo.

## Main Concepts

- **Reasoning Model (RM):** writes the transformation hypothesis.
- **Prediction Model (PM):** predicts output grids from ARC examples, with or
  without the RM's guidance.
- **Test-Time Training (TTT):** creates a temporary adapter for one task using
  augmented versions of that task.
- **Budgeted Evaluation:** reports performance together with the compute spent on
  reasoning, adaptation, and sampling.

## Notes

Large datasets, model weights, checkpoints, run outputs, logs, and local
reference documents are intentionally excluded from Git. Small fixtures may be
added under `tests/fixtures/` when they are needed for tests or documentation.
