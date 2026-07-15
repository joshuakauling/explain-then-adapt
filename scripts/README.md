# Scripts

Scripts are thin command-line entry points. Reusable logic belongs in the
`explain_then_adapt` package.

Implemented:

- `build_reasoning_data.py`: prepares provider-independent generation requests,
  validates results, records automatic or manual acceptance, plans trace-aware
  augmentation, and runs the Gemini or vLLM adapters. Detailed examples live in
  the [data-generation README](../src/explain_then_adapt/data_generation/README.md).
- `migrate_reasoning_resources.py`: deterministically rebuilds the small
  versioned task manifest, hint collection, and few-shot pool from explicit
  legacy source paths while validating the recovered corpus counts.

Planned scope:

- training the Reasoning Model and Prediction Model,
- creating per-task TTT adapters,
- running inference,
- evaluating predictions and budget metrics.
