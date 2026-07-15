# Reasoning Data Generation

This package implements the offline pipeline that turns ARC demonstration pairs
into training examples for the Reasoning Model. The target is a structured
reasoning trace followed by a concise transformation description and a list of
general application steps.

## Inputs

The initial generation stage uses:

- ARC demonstration pairs,
- a pool of manually written few-shot reasoning examples,
- optional manually curated hints.

The small versioned inputs live in
[`resources/data_generation/`](../../../resources/data_generation/): the final
task manifest, the complete hint collection, and the five few-shot traces. ARC
task grids remain a separate local dataset and are supplied through the CLI.

Hints provide weak supervision through five fields: `general`, `inputs`,
`outputs`, `transformation`, and `transformation_steps`. A task is treated as
hint-backed only when all five fields are present and non-empty. Tasks without a
complete hint use the same generation pipeline without the hint block; partial
labels are not silently inserted into prompts.

The manually written hint text is preserved as a historical annotation,
including its original spelling and phrasing. Loading removes surrounding
whitespace but does not rewrite or silently correct the content used for
generation.

Of the 624 tasks used for the final reasoning dataset, 481 were generated with
complete manually curated hints. The remaining 143 tasks were generated directly
without hints and then inspected manually. This second route became practical
with Gemini 3 Flash, whose stronger ARC reasoning reduced the effort compared
with writing additional task-specific hints first.

## Target Format

Every accepted trace follows the same parseable structure:

```text
<think>
1) INPUT ANALYSIS
2) OUTPUT ANALYSIS
3) TRANSFORMATION ANALYSIS
4) STEPS FOR THE TRANSFORMATION
</think>

General natural language description:
...

General steps:
...
```

The reasoning block compares the demonstrations and searches for a consistent
transformation. The two sections after `</think>` provide the compact guidance
later consumed by the Prediction Model.

## Pipeline

### 1. Initial Generation

The final initial traces were generated with Gemini 3 Flash. Every request used
two few-shot examples. The earlier batch fixed the pair `6430c8c4` and
`7c008303`; later retry and local-generation runs used different pairings from a
manually prepared pool of five tasks. The rebuilt records therefore store the
selected task IDs for every request instead of assuming one global pair.

Hint-backed and hint-free generation are recorded as variants of the same stage
rather than as unrelated datasets. Hint-free candidates are marked for manual
inspection before entering the accepted trace pool.

### 2. Static Validation

Generated text is normalized before validation. The static checks require every
tag and heading from the target format in the correct order. Candidates with
missing or malformed sections are rejected.

### 3. Judge Validation

The automatic quality check uses gpt-oss-120b through vLLM as an LLM judge. Five
independent verdicts are collected for each candidate, and automatic acceptance
requires five passes out of five. Judge responses are parsed as JSON rather than
accepted through substring matching.

The hint-free generation route used manual inspection instead of repeating the
full judge procedure when that additional compute was not cost-effective. The
rebuilt pipeline records these decisions as `manual_review`; it does not present
them as `judge_5_of_5` results.

### 4. Trace-Aware Augmentation

Each task is augmented through a combination of:

- one of the eight geometric ARC transformations,
- a permutation of values 0 through 9,
- a permutation of the demonstration order.

The same transformation is applied to every input and output grid. Because the
original trace refers to directions, values, and demonstration indices, an LLM
then rewrites the trace against the transformed task. The rewrite was performed
locally with gpt-oss-120b through vLLM.

Augmentation proceeds in multiple runs until each task has 100 accepted variants.
Malformed rewrites are discarded and replaced by newly generated variants. The
rewrite stage uses static schema validation; it does not run the full five-vote
LLM judge for every augmented trace.

## Execution Backends

The pipeline separates provider-independent records and prompts from execution:

- **Gemini:** remote generation, including asynchronous batch submission and
  result collection.
- **vLLM:** local batched inference for judging and trace rewriting.

Both backends emit the same normalized result record with task identity,
stage, model, prompt version, few-shot keys, sampling parameters, token usage,
raw output, normalized output, and validation provenance.

Provider credentials, local model paths, queue directories, and run-specific
repair scripts do not belong in the reusable package.

## Package Layout

```text
data_generation/
  cli.py
  hints.py
  resource_migration.py
  records.py
  prompts.py
  postprocessing.py
  validation.py
  augmentation.py
  pipeline.py
  backends/
    gemini.py
    vllm.py
```

The legacy five-part augmented keys included a record index. New records should
store source-trace identity and augmentation parameters as explicit fields; the
legacy key encoding should not be carried into the rebuilt pipeline.

## Command-Line Workflow

Install the package in editable mode before using the script. Gemini and vLLM
remain optional so prompt construction, postprocessing, and tests work without
either provider stack:

```bash
python -m pip install -e .
python -m pip install -e '.[gemini]'  # remote generation
python -m pip install -e '.[vllm]'    # local judging and rewriting
```

All CLI stages exchange provider-independent JSONL files. The usual initial
generation route is:

```bash
python scripts/build_reasoning_data.py prepare-initial \
  --tasks-dir /path/to/arc/tasks \
  --task-ids resources/data_generation/task_manifest.jsonl \
  --hints resources/data_generation/hints.jsonl \
  --few-shot-manifest resources/data_generation/few_shot_traces.jsonl \
  --provider-defaults \
  --output /path/to/run/initial.requests.jsonl

python scripts/build_reasoning_data.py gemini-export \
  --input /path/to/run/initial.requests.jsonl \
  --output /path/to/run/gemini.requests.jsonl
```

The few-shot manifest may be the versioned JSONL resource or a JSON list. A
trace can be embedded directly or loaded from a path relative to the manifest.
When the path contains a mapping of task IDs to traces, the entry's `task_id` is
used as the lookup key by default; an explicit `trace_key` may override it:

```json
[
  {"task_id": "example_a", "trace_path": "traces/accepted.json"},
  {"task_id": "example_b", "trace": "<think>..."}
]
```

The historical Gemini 3 Flash requests omitted `generation_config` and therefore
used the model's provider defaults. `--provider-defaults` preserves that behavior
in the exported Gemini JSONL instead of guessing temperature, top-p, thinking,
or output-token settings. Explicit sampling arguments remain available for new
experiments and are always required by the vLLM backend.

After provider results have been imported, the remaining stages are explicit:

```bash
# Normalize and reject malformed traces.
python scripts/build_reasoning_data.py validate-static \
  --input /path/to/run/initial.results.jsonl \
  --output /path/to/run/initial.validated.jsonl

# Create exactly five judge requests for every statically valid trace.
python scripts/build_reasoning_data.py prepare-judge \
  --input /path/to/run/initial.validated.jsonl \
  --tasks-dir /path/to/arc/tasks \
  --hints resources/data_generation/hints.jsonl \
  --output /path/to/run/judge.requests.jsonl

# Execute judge or rewrite requests locally.
python scripts/build_reasoning_data.py vllm-run \
  --input /path/to/run/judge.requests.jsonl \
  --model /path/to/local/model \
  --output /path/to/run/judge.results.jsonl

# Attach a judge_5_of_5 decision to each initial trace.
python scripts/build_reasoning_data.py evaluate-judges \
  --sources /path/to/run/initial.validated.jsonl \
  --judge-requests /path/to/run/judge.requests.jsonl \
  --judge-results /path/to/run/judge.results.jsonl \
  --output /path/to/run/initial.accepted.jsonl
```

Manual decisions use a separate JSONL file with `request_id`, boolean
`accepted`, and a non-empty `reviewer_note`. They are attached with
`apply-manual-reviews` and recorded as `manual_review`, never as judge results.

`prepare-rewrite` reads accepted source traces and plans only the number of
variants still missing from the target. On later runs, pass the accumulated
rewrite request and statically validated result files through
`--existing-requests` and `--existing-results`. Accepted variants count toward
the default target of 100; rejected attempts are excluded from future plans.

For an initial-generation retry, use `--candidate-start` to continue with a new
candidate index instead of regenerating an earlier request identity. For
example, `--candidate-start 1 --candidates-per-task 1` creates only candidate 1.

Run `python scripts/build_reasoning_data.py --help` or the help for an individual
subcommand for the complete argument list. `render-requests` writes each internal
request as a readable text file for prompt review without changing the JSONL
source.

## Data Storage

Full prompts, provider responses, generated traces, and augmented corpora are
generated artifacts and stay outside Git. The small task manifest, curated
hints, and few-shot pool are versioned under `resources/data_generation/` because
they define how generation requests are assembled. A released training corpus
should be published as a versioned dataset instead of copied into the source
tree.
