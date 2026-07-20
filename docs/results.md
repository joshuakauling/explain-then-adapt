# Results

This page reports the final experiments from the master's thesis. These are
historical results, not measurements from a fresh end-to-end rerun of the
rebuilt repository. The original model checkpoints are no longer available, so
new training and inference runs require freshly trained models.

The surviving final Guided prediction artifacts can still be scored. The
rebuilt evaluator reproduces their reported thesis values exactly and adds a
stricter solve metric for tasks with multiple test inputs. This distinction is
made explicit below.

## Evaluation Conventions

The final experiments use two candidate protocols:

| Protocol | Candidates per test input | Candidate construction |
| --- | ---: | --- |
| Standard | 32 | 32 samples from the original task view |
| Aug 64 | 64 | one sample from each of 64 augmented task views |

`Solve` in the thesis is an oracle best-of-N metric: a task counts as solved
when its candidate pool contains a correct answer. The historical evaluator
aggregated by original task ID and therefore counts a multi-test task as solved
when any test input has a correct candidate. It does not represent the accuracy
of a selector that chooses one final submission.

The Test 2024 split contains 337 tasks: 319 have one test input and 18 have two.
The rebuilt evaluator reports both the historical **Thesis Solve** and the
stricter **All-Test-Inputs Solve**, under which every test input of a task must
be covered. `Acc` is sample accuracy over all generated grids.

## Guidance Without TTT

The first experiment isolates the offline Prediction Models and does not apply
task-specific parameter updates.

| Model | Standard Solve | Standard Acc | Aug 64 Solve | Aug 64 Acc |
| --- | ---: | ---: | ---: | ---: |
| Unguided | **23.15%** | **11.14%** | 37.39% | 10.22% |
| Unguided-ReARC | 15.43% | 5.01% | 26.71% | 4.78% |
| Guided-see-first | 17.51% | 8.78% | 37.39% | 7.96% |
| Guided | 18.40% | 10.73% | **39.17%** | 9.76% |
| Guided-ReARC | 21.96% | 10.84% | 38.28% | **11.27%** |

Guidance alone is not consistently beneficial. Unguided has the highest solve
rate under Standard sampling, while Guided has the highest Aug 64 solve rate.
The result does not support the claim that adding a rule description to the
prompt is sufficient by itself.

Masking the first output also matters: Guided outperforms Guided-see-first in
all four reported measures. In this setup, treating the first input-output pair
only as observed context is more effective than also applying loss to its
assistant grid.

## Guidance With TTT

The second experiment adds 64 task-specific TTT updates before candidate
generation. `Guided` and `Guided-ReARC` use augmentation-specific rule
descriptions; `Unguided` adapts only on the demonstration grids.

| Split | Model | Standard Solve | Standard Acc | Aug 64 Solve | Aug 64 Acc |
| --- | --- | ---: | ---: | ---: | ---: |
| Test 2024 | Guided-ReARC | 38.58% | **28.04%** | 60.24% | 25.74% |
| Test 2024 | Guided | **42.14%** | 27.82% | **65.28%** | 26.44% |
| Test 2024 | Unguided | 35.61% | 27.44% | 61.13% | **26.82%** |
| Test 2025 | Guided-ReARC | **1.67%** | **0.53%** | 3.33% | 0.57% |
| Test 2025 | Guided | **1.67%** | 0.35% | **6.67%** | 0.45% |
| Test 2025 | Unguided | 0.83% | 0.07% | **6.67%** | **0.59%** |

On Test 2024, full-budget Guided TTT improves solve over Unguided TTT by 6.53
percentage points under Standard sampling and by 4.15 points under Aug 64. All
three variants improve substantially relative to their no-TTT counterparts,
which also shows that adaptation itself accounts for a large part of the gain.

On the harder 120-task Test 2025 split, the absolute solve counts are small.
Guided and Guided-ReARC solve two Standard tasks each, compared with one for
Unguided. Under Aug 64, Guided and Unguided both solve eight tasks. These values
do not show a consistent advantage for guidance on that split.

## Compute Budget

The guidance budget `k` controls how many of the 64 TTT updates receive an
augmentation-specific explanation. The final candidate count remains fixed at
64 for the budgeted augmented protocol. The reported cost uses the thesis's
A100 seconds-equivalent model, converted to hours; it is not a direct wall-clock
measurement.

| Guidance budget `k` | Cost | Budgeted Aug Solve | Budgeted Aug Acc | Standard Solve | Standard Acc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 131.0 h | **65.28%** | 26.44% | **42.14%** | **27.82%** |
| 32 | 78.7 h | 62.91% | 25.73% | 38.28% | 25.20% |
| 16 | 52.1 h | 60.83% | 25.07% | 40.36% | 25.10% |
| 8 | 37.9 h | 57.27% | 24.15% | 35.91% | 24.23% |
| 0 | 28.5 h | 33.53% | 21.13% | 33.53% | 21.13% |
| Unguided | **26.6 h** | 61.13% | **26.82%** | 35.61% | 27.44% |

On the budgeted augmented protocol, guidance first exceeds the Unguided solve
rate at `k = 32`: 62.91% versus 61.13%. Its estimated cost is 78.7 hours,
compared with 26.6 hours for Unguided. Full guidance reaches 65.28% at 131.0
hours. The extra 4.15 solve points over Unguided therefore cost roughly 4.9
times as much under this model.

On Standard evaluation, `k = 8` narrowly exceeds Unguided solve and `k = 16`
reaches 40.36%. Increasing from `k = 16` to `k = 64` adds only 1.78 solve points
while increasing estimated cost from 52.1 to 131.0 hours. Results are not
strictly monotonic at every intermediate budget, but the upper end clearly
shows diminishing returns.

## Answers to the Research Questions

1. **Does reasoning-guided adaptation beat unguided adaptation?** Yes on Test
   2024 at the full guidance budget: Guided achieves the highest Standard and
   Aug 64 solve rates. The advantage is not universal across splits, metrics, or
   compute budgets.
2. **Is the explanation useful without adaptation?** Not reliably. Unguided is
   strongest under Standard sampling without TTT. The clearest guidance benefit
   appears when the rule is repeated as a training signal during adaptation.
3. **How does performance scale with test-time compute?** More guidance generally
   improves solve coverage, but the gains flatten and are expensive. In the
   tested setup, Explain-then-Adapt is more accurate at high budget but not more
   compute-efficient than Unguided TTT.

## Rebuild Verification

The final Guided artifacts provide the following exact regression target:

| Protocol | Thesis Solve | All-Test-Inputs Solve | Sample Accuracy |
| --- | ---: | ---: | ---: |
| Standard | 42.1365% | 40.9496% | 27.8169% |
| Aug 64 | 65.2819% | 64.9852% | 26.4393% |

The Thesis Solve and Sample Accuracy columns reproduce the rounded values in the
final result table. The lower strict solve rates come from requiring both test
inputs to be covered on multi-test tasks; they do not indicate a change in the
underlying predictions.

The rebuilt code is covered by CPU tests for data contracts, augmentation,
training-loop behavior, inference planning, inverse transformations, parsing,
and metric aggregation. Real model execution has not been rerun because the
historical checkpoints and equivalent GPU hardware are unavailable.

## Limitations

- The reported solve rates measure oracle coverage of a candidate pool. The
  thesis did not train or evaluate a selector that chooses one final candidate.
- The historical checkpoints and the 97,461-row augmented reasoning corpus no
  longer exist. Fresh runs can reconstruct the pipeline, but not byte-identical
  model states or training examples.
- Natural-language guidance can be incomplete or wrong, and the final pipeline
  has no rule verifier before TTT.
- The cost model depends on measured A100 throughput and is intended for
  relative comparisons between these runs.
- The low Test 2025 solve counts make differences between configurations
  sensitive to only a few tasks.

See [Methodology](methodology.md) for the experiment design and the
[evaluation README](../src/explain_then_adapt/evaluation/README.md) for exact
metric and artifact contracts.
