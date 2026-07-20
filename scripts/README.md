# Scripts

Scripts are thin command-line entry points. Reusable logic belongs in the
`explain_then_adapt` package.

Implemented:

- `build_reasoning_data.py`: prepares provider-independent generation requests,
  validates results, records automatic or manual acceptance, plans trace-aware
  augmentation, and runs the Gemini or vLLM adapters. Detailed examples live in
  the [data-generation README](../src/explain_then_adapt/data_generation/README.md).
- `migrate_reasoning_resources.py`: deterministically rebuilds the versioned
  task manifest, hint collection, few-shot pool, selected 624-task base-trace
  set, and format-repair audit from explicit legacy source paths.
- `migrate_reasoning_training_resources.py`: migrates and verifies the raw and
  augmented 39-task Reasoning Model validation views.
- `build_reasoning_training_data.py`: joins accepted rewrite results with their
  structured augmentation requests and builds an external, tokenizer-specific
  training cache plus a readable provenance manifest.
- `train_reasoning_model.py`: runs the final thesis-aligned task-balanced QLoRA
  training loop, dual-view validation, and checkpoint schedule.
- `build_prediction_training_data.py`: builds guided or unguided synthetic PM
  caches from accepted rewrites, or a separately provenance-tracked ReARC cache.
- `train_prediction_model.py`: resolves and trains one of the five final
  Prediction Model profiles with exact multi-assistant loss masking.
- `merge_lora_adapter.py`: merges an adapter into a standalone base model; this
  is the explicit handoff between Unguided-ReARC and Guided-ReARC training.
- `train_ttt_adapters.py`: creates one fresh 64-update LoRA adapter per selected
  ARC task, with deterministic augmentation plans, nested guidance budgets,
  atomic task manifests, and resume support.
- `build_inference_plan.py`: creates the canonical 64-variant plan shared by RM
  sampling, online TTT, and PM sampling.
- `sample_reasoning_model.py`: samples exactly one RM trace per required original
  or augmented task view and writes validated guidance plus token accounting.
- `run_prediction_inference.py`: implements `standard32`, `augmented64`, and
  `budgeted64` PM candidate generation with optional per-task TTT adapters.
- `evaluate.py`: validates structured PM artifacts, inverse-transforms
  augmented grids, and writes thesis-compatible plus strict multi-test metrics.
- `summarize_compute.py`: combines provenance-linked RM, TTT, and PM token
  counts into the thesis seconds-equivalent cost measure.
