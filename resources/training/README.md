# Training Resources

This directory contains the small, versioned resources used for raw and
augmented validation during both offline model-training stages. It does not
contain token caches, model weights, generated training variants, or ARC task
grids.

## Files

- `reasoning_validation.jsonl`: one manually reviewed reasoning trace for each
  of the 39 held-out ARC-AGI-1 evaluation tasks used as the raw validation view.
- `reasoning_validation_augmented.jsonl`: one manually reviewed, structured
  augmentation of each of the same 39 tasks.

The validation tasks do not overlap the 624-task Reasoning Model training
manifest. All 78 source traces pass the current strict trace-format validator.

These resources are tokenized again by `scripts/build_reasoning_training_data.py`
and `scripts/build_prediction_training_data.py`. The old `.pt` caches are
deliberately not migrated because generated token caches should be tied to an
explicit tokenizer, format, and revision.
