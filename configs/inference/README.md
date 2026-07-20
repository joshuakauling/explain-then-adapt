# Inference Configuration

`inference.yaml` contains the final sampling, serialization, and vLLM defaults
used by the rebuilt inference pipeline. Model paths, task paths, adapter paths,
and output locations remain explicit CLI arguments because they are local run
state rather than versioned experiment settings.

The three protocols correspond directly to the thesis evaluations:

- `standard32`: the original task, sampled 32 times per test input;
- `augmented64`: all 64 augmentation-plan variants, sampled once each;
- `budgeted64`: for `k > 0`, the selected `k` variants are each sampled
  `64 / k` times; for `k = 0`, the original task is sampled 64 times.

The augmentation seed is shared with Test-Time Training. Build one plan and
pass the same file to Reasoning Model sampling, TTT, and Prediction Model
sampling.
