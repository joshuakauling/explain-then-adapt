# Evaluation

This stage turns raw Prediction Model completions into exact-match metrics and
accounts for the test-time compute used by RM sampling, TTT, and PM sampling.
It does not select or rerank candidates; the thesis evaluates the generated
candidate pool with oracle best-of-N solve and sample accuracy.

## Input Contract

`evaluate.py` accepts the JSONL artifact written by
`run_prediction_inference.py` and requires its sibling manifest. Before scoring,
it verifies:

- the JSONL SHA-256 digest and schema,
- the selected task IDs and hashes of their labelled JSON files,
- unique request IDs and consistent variant metadata,
- complete coverage of every test input,
- the declared request and candidate budgets.

The task directory must contain labelled test outputs. Public ARC test tasks
without solutions cannot be scored locally.

## Grid Scoring

The parser removes Qwen chat-boundary tokens and an exact leading `assistant`
line, then reads consecutive digit-only rows from the start of the completion.
Trailing prose is ignored, as in the research evaluator. Empty, ragged,
non-leading, or larger-than-30-by-30 grids are recorded as parse failures and
count as incorrect samples.

Predictions from augmented protocols are emitted in the variant's geometry and
value space. Evaluation first reverses the geometry and then the value
permutation, producing a canonical grid in the original task space. Exact match
is performed there. Demonstration-order permutations do not affect the output
grid and therefore require no inverse operation.

## Metrics

- **Thesis Solve:** fraction of original task IDs with at least one correct
  candidate for at least one test input. This intentionally reproduces the
  `Orig-Key` aggregation used for the reported thesis tables.
- **All-Test-Inputs Solve:** fraction of original tasks for which every test
  input has at least one correct candidate. This additional metric exposes the
  stricter behavior for tasks with multiple test inputs.
- **Test-Input Solve:** fraction of individual test inputs covered by at least
  one correct candidate.
- **Request Solve:** fraction of variant/test-input requests with at least one
  correct sample.
- **Sample Accuracy:** correct output grids divided by all generated output
  grids.
- **Parse Success:** valid parsed grids divided by all generated outputs.

The ARC-2024 thesis split contains 337 tasks, including 18 with two test inputs,
so Thesis Solve and All-Test-Inputs Solve are not interchangeable. Both are
always written to the summary.

## Running Evaluation

Install the lightweight evaluation dependency once:

```bash
python -m pip install -e '.[evaluation]'
```

```bash
python scripts/evaluate.py \
  --config configs/evaluation/evaluation.yaml \
  --predictions runs/inference/guided_budget64/predictions.jsonl \
  --tasks-dir /path/to/labelled/arc_tasks \
  --output-dir runs/evaluation/guided_budget64
```

The output directory is published only after successful validation and
contains:

- `summary.json`: aggregate metrics, definitions, configuration, and hashes;
- `report.md`: a compact human-readable table for the evaluated run;
- `tasks.jsonl`: expected outputs and per-task/per-test-input counts;
- `candidates.jsonl`: parse status, inverse-transformed grid, and exact-match
  result for every generated candidate.

Existing output directories are never overwritten. A failed run leaves a
visible `.partial` directory for inspection.

## Compute Accounting

The thesis cost model is:

```text
C_total = (T_prefill + 3 * T_train) / r_prefill + T_gen / r_decode
```

Reasoning and Prediction Model prompt tokens contribute to `T_prefill`, RM
traces and PM completions contribute to `T_gen`, and all sequence tokens
processed by TTT contribute to `T_train`. The checked-in A100 profile uses
5,000 prefill tokens/s and 75 decode tokens/s. The result is a
seconds-equivalent approximation rather than a hardware-independent measure.

```bash
python scripts/summarize_compute.py \
  --config configs/evaluation/evaluation.yaml \
  --predictions runs/inference/guided_budget64/predictions.jsonl \
  --reasoning runs/inference/guided_budget64/reasoning.jsonl \
  --ttt-run runs/ttt/guided_budget64 \
  --output runs/evaluation/guided_budget64/compute.json
```

Repeat `--reasoning` when PM inference and TTT used different RM artifacts.
Their hashes are matched against both manifests, and an artifact shared by both
stages is counted once. Omit `--ttt-run` for non-TTT inference and omit
`--reasoning` only when neither PM nor TTT used guidance.
