# Reasoning-Generation Resources

This directory contains the human-readable resources needed to prepare
reasoning-generation and trace-rewrite requests. It includes the selected base
trace for every training task, but not ARC grids, provider responses, or the
large augmented training corpus.

## Files

- `base_reasoning_traces.jsonl`: one selected reasoning trace for each of the
  624 training tasks. Every trace passes the current strict format validator.
- `base_trace_repairs.jsonl`: provenance for 24 format-only repairs applied
  during migration. Twenty-two records received a missing `</think>` tag, and
  two had a duplicated compact summary removed from inside the reasoning block.
- `hints.jsonl`: 481 complete, manually curated five-field hints. Original
  spelling and phrasing are preserved; only surrounding whitespace was removed.
- `few_shot_traces.jsonl`: the five accepted traces used as the later few-shot
  selection pool, kept in historical pool order.
- `task_manifest.jsonl`: one record for each of the 624 final reasoning tasks.
  It records the historical corpus partition, raw hint completeness, generation
  hint mode, few-shot membership, accepted-trace availability in the legacy
  source, and recoverable validation provenance.

The manifest's `legacy_training_2024` and `training_2025_addition` labels refer
to the names and set relationship of the recovered project subsets. They should
not be interpreted as a reconstructed claim about an official upstream split.

For the 143 hint-free tasks, manual inspection is recorded from the documented
historical workflow. A task-level automatic validation route could not be
recovered reliably for the 481 hint-backed tasks, so those records use
`historical_validation_route: "unknown"` rather than an inferred 5/5 result.

## Regeneration

These files were created with `scripts/migrate_reasoning_resources.py` from:

- the final 624-task ID list,
- its 391-task legacy subset,
- the LabelingARC hint files,
- the final task-to-accepted-trace mapping.

The migration command requires all source paths explicitly and validates the
expected counts and ID coverage before replacing the resources. Legacy sources
remain local and excluded from Git.

The historical 97,461-row augmented corpus is not available and is not part of
this repository. To build a new corpus, use `base_reasoning_traces.jsonl` as the
source for `prepare-rewrite` and run the resulting requests through a supported
inference backend. Generated requests, responses, and token caches remain
outside Git. Older surviving augmentation files are not included because their
source traces cannot be matched reliably to this final base-trace selection.
