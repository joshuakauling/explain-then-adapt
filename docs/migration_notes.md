# Migration Notes

This document records evidence recovered from the legacy `ARC/` reference tree
and the decisions made while rebuilding the corresponding workflow. Legacy
paths and run artifacts remain excluded from Git.

## Reasoning-Data Generation

### Historical Gemini Configuration

The preserved Gemini response files contain 2,395 successfully parsed response
records. Every record reports the same model version:

```text
gemini-3-flash-preview
```

The original batch builders and the preserved `rest_no_labels.jsonl` request
file do not include `generation_config`. Initial generation therefore relied on
the model's provider defaults rather than a recorded temperature, top-p,
thinking level, or output-token limit. The rebuilt Gemini exporter represents
this explicitly through `SamplingParameters(use_provider_defaults=True)` and
omits `generation_config` instead of guessing missing values.

As of July 2026, Google's
[model lifecycle documentation](https://ai.google.dev/gemini-api/docs/deprecations)
still lists `gemini-3-flash-preview` without a shutdown date and recommends
`gemini-3.5-flash` as its eventual replacement. The
[Gemini 3 guide](https://ai.google.dev/gemini-api/docs/gemini-3) also confirms
Batch API support. Reproduction runs should use the historical model ID while it
remains available; experiments with a replacement model must be recorded as a
separate configuration.

### Few-Shot Selection

Every initial-generation request contains two worked examples. The earliest
Gemini batch fixed the pair:

```text
6430c8c4
7c008303
```

Later retry and local-generation runs selected pairs from this five-task pool:

```text
6430c8c4
7c008303
08ed6ac7
60b61512
f25fbde4
```

The rebuilt selector preserves the legacy pool order and reproduces its
SHA-256-based per-task seed derivation. Selected few-shot IDs are stored in every
request and result record.

### Hint Routes

The final 624-task reasoning set contains 481 tasks with complete manual hints
and 143 tasks generated without complete hints. A hint is complete only when all
five required fields are non-empty. Partial hints are not inserted into prompts.

Original spelling and phrasing are preserved as historical annotations. The
hint-free route contains no target hint block and is recorded separately for
manual review.

The versioned migration produced 481 complete hint records and a 624-task
manifest. Of the 143 hint-free tasks, 91 had no source label file and 52 had a
source file missing at least one required field. The historical corpus is also
recorded as 391 IDs from the recovered `training_2024` subset plus 233 later
additions. These names describe the recovered project subsets, not an inferred
official ARC release split.

### Intentional Rebuild Differences

The legacy Gemini requests placed the role instruction and task prompt together
in one user message. The rebuilt backend uses a system instruction plus one user
message while preserving the task content. This is an intentional interface
cleanup, not a claim that the byte-level historical request was reproduced.

Legacy augmented identifiers encoded transformation fields and, in some runs, a
record index in one string. Rebuilt records store source trace, geometry, value
mapping, example order, style, and variant index as explicit fields.

## Pilot Status

A six-task pilot covers three hint-backed and three hint-free tasks. Its
internal requests, readable prompts, provider responses, and validation results
are stored under the ignored `data/pilot/` directory. The exported requests use
the historical model defaults and contain no `generation_config`.

The batch was submitted to `gemini-3-flash-preview` on July 14, 2026 and
completed successfully with all six responses. Five traces passed the strict
static schema validation. The response for task `253bf280` contained a plausible
transformation description but omitted the four required analysis headings
inside `<think>`, so the pipeline correctly rejected it as malformed.

The failed task was regenerated as candidate 1 with a newly selected few-shot
pair, without resubmitting candidate 0. The retry passed the static schema and
its connection rule reproduced all eight demonstrations. After explicit manual
review, the pilot therefore closed with six accepted task traces and one
discarded malformed attempt.

The optional SDK was installed in an isolated test environment with
`google-genai==1.47.0`. Credentials are loaded from the ignored local `.env`
file and are not stored in run artifacts.

The temporary verification environment uses Python 3.9, for which current
Google authentication packages emit an end-of-life warning. This does not affect
request serialization, but the supported Python baseline should be revisited
before provider-backed runs become part of the documented public setup.

## Reasoning-Model Training Audit

The thesis, final legacy configuration, notebook metadata, and checkpoint naming
agree on the following training contract:

- base model: `Qwen/Qwen3-4B-Thinking-2507`;
- QLoRA with 4-bit base weights and LoRA rank 128, alpha 32, and no dropout;
- adapters on all attention and feed-forward projection layers;
- assistant-only cross-entropy loss over the complete trace, starting at
  `<think>` and including the compact final description;
- no system message and no test pair in the Reasoning Model training prompt;
- 100 epochs, one distinct augmented variant per task and epoch;
- micro-batch size 2 with eight gradient-accumulation steps;
- maximum sequence length 8,192 tokens;
- cosine learning-rate schedule from `1e-4` to `2e-5` with 5% warmup;
- validation every 80 optimizer steps and checkpoints every 400 steps.

The recovered pre-tokenization metadata reports 97,461 variants across 624
tasks. Applying the recorded 8,192-token limit retained 97,449 variants and all
624 tasks, with at least 100 variants available per task. This is consistent
with 39 optimizer steps per epoch and the later checkpoint names: step 780 after
20 epochs and step 1,170 after 30 epochs.

The current legacy YAML is not itself a reproduction configuration. It still
contains the constant learning rate used by an earlier sweep, while the thesis
records the schedule selected for the final run. Likewise, the defaults in
`ARC/build_dataset_RM.py` would remove the reasoning block by slicing after
`</think>`. Notebook metadata confirms that the actual cache instead used
`marker="<think>"`, `keep_marker=True`, and `skip_if_missing_marker=True`.

The final `reasoning_training_data.jsonl`, `ids_by_key.pt`, and
`ids_by_key_max8192.pt` artifacts are not present in the recovered local clone or
its Git history. The three postprocessed augmentation directories referenced by
the notebook are also absent. The 624 accepted base traces remain available,
but the complete set of rewritten augmented traces cannot be reconstructed
byte-for-byte from the local files. Before migrating the training implementation,
the project must therefore choose one canonical source:

1. recover the final JSONL or token cache from the original cluster/NFS storage;
2. regenerate a new, explicitly versioned corpus with the rebuilt data pipeline.

The resulting large corpus and token cache remain external artifacts and must
not be committed to Git.
