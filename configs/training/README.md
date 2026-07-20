# Training Configurations

`reasoning_model.yaml`, `prediction_model.yaml`, and `test_time_training.yaml`
record the final thesis configurations for offline fine-tuning and online
adaptation. They contain portable model, data, optimizer, scheduler, loader,
checkpoint, and profile settings; local paths remain CLI arguments.

The legacy Reasoning Model YAML is not used because it records an earlier
learning-rate experiment. The final thesis specifies the selected cosine
schedule, 100-epoch protocol, and validation cadence represented here.

The historical 400-step checkpoint interval does not produce the later
candidate checkpoints at steps 780 and 1,170. The configuration therefore keeps
the documented periodic interval and records epochs 20 and 30 as explicit
checkpoint boundaries. The final adapter is saved separately after epoch 100.

Generated token caches embed the resolved configuration and source checksums.
Changing the tokenizer, sequence limit, variant target, or prompt formatting
therefore requires rebuilding the cache rather than reusing an old `.pt` file.

The Prediction Model file resolves five named profiles from one shared setup:
`guided`, `guided_see_first`, `unguided`, `unguided_rearc`, and
`guided_rearc`. Guided profiles share one token cache, while unguided synthetic
data and external ReARC data each require their own cache. Profile-specific
fields make guidance, first-response masking, initialization, duration, and
candidate checkpoints visible without duplicating the common QLoRA settings.

At 39 updates per synthetic epoch, the recovered epoch-40 and epoch-60
candidates are steps 1,560 and 2,340. The `1570` label found in part of the
thesis is inconsistent with both this arithmetic and the thesis evaluation
table, so the configuration records epoch 40. Guided also preserves the
off-interval step-2,500 candidate. Periodic 400-step checkpoints and final
adapters remain enabled for every profile.

The Test-Time Training file resolves `guided` and `unguided` profiles. Guided
runs accept nested budgets of 0, 8, 16, 32, or 64 augmentation-specific
explanations and use a peak learning rate of `1.25e-4`; unguided runs omit the
system message and use `2e-4`. Both profiles create a fresh rank-32 adapter for
each task, train for exactly 64 updates, warm up for 32 updates, and decay to
`5e-6` over the remaining 32.

The one-space `empty_guidance_content` value is part of the experiment contract,
not cosmetic YAML whitespace. It preserves the guided prompt shape for variants
outside a partial guidance budget. Truly missing selected guidance follows the
separate `missing_guidance_policy` and fails by default.
