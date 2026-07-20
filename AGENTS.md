# AGENTS.md

## Purpose

This repository is the clean rebuild of the ARC project. The existing `ARC/`
directory is a legacy clone and should be treated as a reference source, not as
the final project layout.

The repository should be easy for external contributors and reviewers to
navigate. Its structure should make the project goal, training pipeline, and
main implementation areas easy to understand quickly.

## Decision Discipline

Do not create, move, refactor, or delete code before the relevant open questions
have been answered. If requirements, target structure, naming, data ownership, or
runtime assumptions are unclear, stop and clarify them first.

Prefer an explicit plan over premature implementation.

## Legacy Source Rules

- Treat `ARC/` as a read-only reference unless explicitly instructed otherwise.
- Do not commit the `ARC/` directory directly.
- Do not preserve the old structure blindly.
- Inspect files before copying them into the new project.

## Repository Safety

- Do not add large datasets, model weights, checkpoints, logs, generated
  experiment outputs, or local environment files to Git.
- Do not delete, move, or rewrite history without explicit confirmation.
- Check `git status` before and after larger cleanup or migration steps.

## Migration Workflow

1. Inspect the legacy source before copying anything.
2. Classify files as: keep, refactor, archive, or ignore.
3. Clarify the target location and naming before moving code.
4. Copy only files that have a clear role in the rebuilt project.
5. Normalize imports, paths, CLI arguments, and configuration handling during
   migration.
6. Verify each migrated piece with the smallest useful check.

## Target Layout

Use a pipeline-oriented layout that foregrounds the training and evaluation
workflow while keeping reusable code in a proper Python package:

```text
README.md
AGENTS.md
configs/
  data_generation/
  training/
  inference/
  evaluation/

scripts/
  build_reasoning_data.py
  train_reasoning_model.py
  train_prediction_model.py
  train_ttt_adapters.py
  run_inference.py
  evaluate.py

src/
  explain_then_adapt/
    arc/
      # ARC loading, formatting, task utilities, and transformations.

    data_generation/
      # Reasoning trace generation, augmentation, validation, and postprocessing.

    training/
      # Reasoning model training, prediction model training, TTT adapter training,
      # LoRA utilities, collators, and shared training code.

    inference/
      # Reasoning sampling, prediction sampling, TTT inference, adapter handling,
      # and vLLM runners.

    evaluation/
      # Prediction parsing, scoring, solve accuracy, budget metrics, and reports.

docs/
  methodology.md
  migration_notes.md
  results.md

notebooks/
  exploration/

tests/
  fixtures/
```

Prefer this layout unless a later explicit decision changes it. Do not recreate
the old `ARC/` structure inside the new project.

## Coding Standards

- Avoid side effects at import time.
- Prefer explicit CLI arguments over hard-coded local paths.
- Keep reusable logic out of notebooks when it becomes part of the project.
- Prefer structured configuration over scattered constants.

## Verification

For code changes, run the smallest relevant verification step before finishing.
If no reliable verification exists yet, state that explicitly.
