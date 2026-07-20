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

Legacy notebook metadata reports 97,461 variants across 624 tasks. Applying the
recorded 8,192-token limit retained 97,449 variants and all 624 tasks, with at
least 100 variants available per task. These counts describe an unavailable
historical artifact; they are not counts derived from files distributed with
this repository. They remain consistent with 39 optimizer steps per epoch and
the later checkpoint names: step 780 after 20 epochs and step 1,170 after 30
epochs.

The surviving YAML and thesis both state a periodic checkpoint interval of 400
optimizer steps. That interval cannot produce checkpoints 780 and 1,170 and was
an identified mistake in the historical configuration. The rebuilt trainer does
not conceal this discrepancy: it retains the documented 400-step interval and
also saves explicit epoch-boundary candidates after epochs 20 and 30. The final
adapter uses the original zero-based directory name `end_epoch_99`.

The current legacy YAML is not itself a reproduction configuration. It still
contains the constant learning rate used by an earlier sweep, while the thesis
records the schedule selected for the final run. Likewise, the defaults in
`ARC/build_dataset_RM.py` would remove the reasoning block by slicing after
`</think>`. Notebook metadata confirms that the actual cache instead used
`marker="<think>"`, `keep_marker=True`, and `skip_if_missing_marker=True`.

The final `reasoning_training_data.jsonl`, `ids_by_key.pt`, and
`ids_by_key_max8192.pt` artifacts are not present in the recovered local clone or
its Git history and are treated as unavailable. The repository does not offer a
download or claim that these records were recovered. A user who needs the
augmented corpus must generate a new, explicitly versioned run from the selected
base traces through the rebuilt inference pipeline.

The resulting large corpus and token cache remain external artifacts and must
not be committed to Git.

The raw legacy validation JSONL files contain 39 original and 39 augmented
traces, and all 78 pass the current strict format validator without repair.
Three traces appeared malformed only after decoding an old `.pt` cache whose
preprocessing had altered the chat-template boundary. This was a cache artifact,
not a defect in the source traces. The old caches are therefore discarded and
both validation views are rebuilt from the raw JSONL resources with the
configured tokenizer.

The rebuilt optimization loop intentionally follows `ARC/training_RM.py` rather
than replacing it with a trainer framework. Model loss is divided by the
configured accumulation count, gradients are accumulated and clipped, and each
optimizer update is followed by the learning-rate scheduler exactly as in the
research implementation. The original `LinearLR` plus `CosineAnnealingLR`
composition and W&B metric names are retained.

Two metric defects are corrected explicitly. First, the original rolling
training-loss tensors were recreated at every epoch boundary. Since an epoch has
39 optimizer steps while logging occurs every five steps, this dropped steps
36--39 from the reported curve and made the step-40 value cover only one update.
The rebuilt window persists across epochs. Second, validation now aggregates
cross-entropy by assistant-target token count; the original implementation gave
every batch equal weight, including the final one-sample batch.

## Prediction-Model Training Audit

The final thesis is again the source of truth for the Prediction Model format.
The recovered `ARC/build_dataset_PM.py` and `ARC/training_PM.py` provide matching
implementation evidence but also contain stale defaults and non-portable data
paths that are not promoted into the rebuilt configuration.

For synthetic PM training, each accepted rewritten trace is joined to the same
structured grid augmentation used for Reasoning Model training. The compact
content after the final `</think>` becomes a system message in guided profiles.
The full chain of thought is not included. Every labelled demonstration and test
pair is transformed, mixed into one random order, and represented by one user
input-grid turn and one assistant output-grid turn. Unguided profiles omit only
the system message.

The legacy builder manually assembled Qwen chat blocks before raw tokenization.
This was necessary because applying the Thinking model's chat template could
insert thinking scaffolding into assistant turns whose targets should contain
only grids. The rebuilt builder retains manual serialization but records exact
assistant-content spans. This makes both PM losses explicit: the default masks
the first output grid, while `guided_see_first` includes it. Chat markers are no
longer selected indirectly by a dynamic header-search collator.

The final experiment contains five PM variants:

- Guided, Guided-see-first, and Unguided use 624 synthetic tasks for 100 epochs,
  giving 39 updates per epoch and 3,900 total updates;
- Unguided-ReARC uses 400 ReARC task types for 100 epochs, giving 25 updates per
  epoch and 2,500 total updates;
- Guided-ReARC starts from the merged final Unguided-ReARC model and applies a
  fresh guided LoRA fine-tune for 40 epochs.

The historical 400-step checkpoint interval misses several candidates used in
the thesis. The rebuilt profiles therefore add only the explicit candidates
required by the reported evaluation: epoch 40 (step 1,560), epoch 60 (step
2,340), and Guided step 2,500. Parts of the prose and result table label the
epoch-40 checkpoint as `1570`, while another evaluation table and recovered run
paths use `1560`. Since `624 / 16 = 39` updates per epoch, step 1,560 is the
consistent value and is encoded as an epoch boundary.

No historical PM training cache survives in the local clone. The public
workflow therefore rebuilds guided and unguided caches from fresh accepted
rewrite records and the versioned 39-task validation traces. It does not decode
or publish the unrelated legacy `.pt` files that were recovered elsewhere in
the tree.

ReARC remains an optional external source. The recovered local `data/re_arc`
directory is absent, so exact identity with the historical export cannot be
verified. The rebuilt source contract pins
`michaelhodel/re-arc` at commit
`e5b7f1d06362a76f9d3b8c25154ff1fafca897ce`, expects 400 JSON files with 1,000
pairs each, and stores a digest of the supplied directory. Packing retains the
legacy maximum of six pairs under the exact token limit and applies independent
geometry and value permutations per pair. The old script's process-dependent
`hash(task_id)` seeding is replaced by stable SHA-256-derived seeds, and its
stale default of 80 variants is replaced by the final 100-epoch contract.

Recovered configuration evidence confirms that Guided-ReARC did not continue
the same LoRA parameterization. `rearc_plus_explain.yaml` points its model name
to a merged Unguided-ReARC model, after which `training_PM.py` attaches a fresh
rank-128 adapter. The rebuilt workflow therefore exposes this merge as an
explicit script and requires the merged directory for the `guided_rearc`
profile.

The PM optimization loop stays close to the research implementation and shares
the same narrow metric corrections as the RM loop: token-weighted validation,
a rolling loss window that survives epoch boundaries, and correct scaling of a
final partial accumulation group. It also rejects malformed source records,
missing task variants, and cache/profile mismatches instead of silently
continuing.

## Test-Time Training Audit

The final thesis defines the online adaptation contract. The recovered
`training_TTT_new.py`, its later YAML files, and evaluation scripts provide
largely matching implementation evidence, but they also retain branches from
earlier experiments. The rebuilt stage implements only the final protocol.

Each ARC task produces 64 training conversations: eight geometric transforms
with eight independently sampled value and demonstration-order permutations per
transform. Only labelled training demonstrations are transformed. One fresh
rank-32, alpha-16 LoRA adapter is attached to the merged Prediction Model for
each task, trained for one batch-size-one epoch, saved, and removed before the
next task. rsLoRA is disabled, matching the later final configuration rather
than the stale `use_rslora=True` branch in the legacy script.

The prompt uses the same manual Qwen chat serialization as Prediction Model
training. Exact assistant-content spans replace the historical fixed-string
search and `prompt[:-10]` trimming. The first assistant output is masked; every
later demonstration output contributes to the loss. This leaves system content,
user grids, chat markers, and padding outside the objective.

Guided runs prepend each selected augmentation's compact natural-language rule
and general steps. Budgets 8, 16, 32, and 64 select 1, 2, 4, or 8 variants from
every transform block using nested evenly spaced indices; budget zero selects
none. An unselected variant still receives a system message containing exactly
one space, preserving the guided PM prompt structure. A genuinely missing RM
output instead omits the system message only under the explicit historical
compatibility policy. The default public workflow treats missing selected
guidance as an error. The unguided baseline always omits the system message and
does not consult guidance data.

The final optimizer settings are AdamW 8-bit, beta values `(0.9, 0.999)`, epsilon
`1e-8`, zero weight decay, and gradient clipping at 1.0. Guided runs peak at
`1.25e-4`; unguided runs peak at `2e-4`. Both warm linearly from zero for 32
updates and then decay by cosine to `5e-6` at update 64. The recovered YAML
contained this floor as `eta_min`, but the legacy Hugging Face scheduler call
never consumed it. The rebuilt scheduler applies the thesis value explicitly.

The historical master augmentation plan was produced with `SystemRandom`, so it
cannot be reconstructed from a seed. The new runner can load that plan when it
is available or create a fresh deterministic plan using SHA-256-derived seeds
that do not depend on task ordering or run names. Generated plans are persisted
with each run. The old branch that reused one original-task explanation across
all augmentations is excluded because it is not part of the final method.

No historical TTT adapters or compatible local GPU setup survive, so numerical
reproduction is not claimed. The rebuilt CPU tests verify the data, prompting,
masking, budget, scheduler, adapter-lifecycle, manifest, and resume contracts.

## Inference Audit

The final inference contract is reconstructed from the thesis together with
`build_inference_data_RM.py`, `inference_RM.py`,
`build_inference_data_PM.py`, `inference_PM.py`, and the later batched TTT
runner. The old scripts mixed hard-coded paths, prompt building, model loading,
sampling, and partially incompatible output formats. The rebuilt pipeline keeps
the same model-facing behavior but separates a deterministic plan, RM guidance,
online TTT, and PM candidates into inspectable artifacts.

RM prompts contain the transformed demonstration pairs in one user message and
are rendered by the model's chat template. PM prompts retain the manual Qwen
serialization used during training: optional compact guidance in a system
message, all transformed demonstration turns, the transformed test input, and
an open assistant header. PM generation stops at either chat boundary token.

Historical RM runs sometimes generated eight traces per original task, but all
surviving downstream code hard-coded `rm_index=0`. The thesis budget accounting
also uses one explanation per required task view. New inference therefore
samples exactly one RM trace instead of generating seven unused alternatives.

The three named PM protocols reproduce the thesis tables directly:
`standard32` draws 32 candidates from the original task, `augmented64` draws one
candidate from each of 64 variants, and `budgeted64` draws `64/k` candidates
from each selected variant for positive `k`. Budget zero uses one original-task
explanation and draws 64 candidates from the original PM prompt. The same
persisted augmentation plan and nested variant indices are consumed by RM, TTT,
and PM.

The legacy inference scripts could silently skip missing guidance or adapters,
or fall back to the base model. The public runner validates all exact guidance
keys and complete 64-update adapter manifests before loading the PM. Large run
outputs are streamed to JSONL with sidecar manifests and exact prompt and
generation token counts, preserving the inputs needed by the thesis cost model.

Inference stores raw outputs in the coordinate and value space of their task
variant. Parsing, inverse transformation, scoring, and task-level aggregation
remain an explicit evaluation responsibility. No candidate selector is added:
the thesis reports best-of-N solve and sample accuracy and identifies selection
or verification as future work.

The original merged checkpoints and suitable GPU hardware are unavailable, so
real vLLM execution cannot be repeated locally. CPU tests cover all request
budgets, exact prompt serialization, JSONL compatibility with TTT, strict
preflight behavior, and a fake-vLLM budgeted run through a per-task adapter.

## Evaluation Audit

The recovered evaluator extracts a leading digit-only grid, compares it with
the labelled output in the same augmented coordinate system, and reports
sample accuracy plus an `Orig-Key` solve rate. Because geometry and value
permutations are bijective, the rebuilt evaluator can instead inverse-transform
each prediction and compare in the original task space without changing the
thesis result. This also makes every saved candidate directly interpretable
against the source ARC task.

The historical `Orig-Key` aggregation marks a task solved when any candidate
for any of its test inputs is correct. This is the metric described and reported
as Solve in the thesis, so it remains available under the explicit name
`thesis_solve`. It is not silently redefined. The ARC-2024 split used here has
337 original tasks: 319 have one test input and 18 have two. The rebuilt report
therefore also includes `all_test_inputs_solve`, which requires candidate
coverage for every test input, as well as test-input and request-level rates.

A temporary local conversion of the surviving, non-distributed final Guided-TTT
predictions verified numerical compatibility during the rebuild. On
`standard32`, the rebuilt evaluator matched 42.14% Thesis Solve and 27.82%
sample accuracy; on `augmented64`, it matched 65.28% and 26.44%. The corresponding
all-test-input solve rates are 40.95% and 64.99%, respectively. The source
prediction artifacts are not included in the repository, so this exact check is
not externally rerunnable. These supplementary values are not substituted into
the historical thesis tables.

All generated candidates remain in the denominator for sample accuracy;
malformed outputs count as incorrect and receive a stable parse status. No
post-hoc candidate selector, verifier, majority vote, or top-two submission
heuristic is introduced.

The compute report implements the final thesis equation directly. RM and PM
prompt tokens form the prefill term, RM and PM generated tokens form the decode
term, and processed TTT sequence tokens receive the training multiplier of
three. It validates the PM, RM, and TTT provenance hashes before combining them,
and deduplicates one RM artifact when it was shared by TTT and final sampling.
The checked-in rates are the measured A100 assumptions from the thesis; reports
using other rates remain structurally valid but are not numerically comparable
without relabelling the hardware profile.

## Selected Base Traces

The legacy `z_gemini/gemini_batch_data/cot/624_best.json` file contains exactly
one selected trace for each of the 624 task IDs in the final training manifest.
It is the semantic source used when asking a model to rewrite a trace against a
geometrically and symbolically transformed task.

Of these traces, 600 already passed the rebuilt strict format validator. The
remaining 24 had narrow serialization defects:

- 22 were missing `</think>` immediately before the compact description;
- 2 repeated the compact description and general steps inside `<think>` before
  presenting them again in the correct location.

Migration inserts only the missing closing tags and removes only the duplicated
inner summaries. It does not substitute alternative candidate answers or edit
the reasoning content. The resulting 624 strict-valid records are versioned as
`resources/data_generation/base_reasoning_traces.jsonl`; the affected task IDs
and repair types are recorded separately in `base_trace_repairs.jsonl`.

## Excluded Legacy Augmentations

Other surviving augmented files were not migrated. Their source traces cannot
be matched reliably to the final `624_best.json` selection and likely belong to
earlier initial-generation runs. Mixing them with the selected base traces would
give the rebuilt corpus unclear provenance.

The public workflow therefore starts exclusively from
`resources/data_generation/base_reasoning_traces.jsonl`. It creates fresh
augmentation specifications, transforms the ARC demonstrations, and rewrites
each trace through the configured inference backend. These newly generated
requests, responses, and accepted variants are run artifacts and remain outside
Git.
