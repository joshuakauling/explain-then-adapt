# Configurations

Versioned experiment settings belong here once the corresponding pipeline stage
has a stable configuration schema. Secrets, machine-local paths, model weights,
and generated run state do not belong in these files. The final Reasoning Model,
Prediction Model, and Test-Time Training configurations are available under
[`training/`](training/README.md).
The final candidate-generation settings and three thesis inference protocols
are available under [`inference/`](inference/README.md).
Parser limits and the A100 throughput assumptions used by the thesis cost model
are available under [`evaluation/`](evaluation/README.md).

Scope:

- default settings for each pipeline stage,
- model and adapter paths,
- data paths and split names,
- sampling parameters,
- TTT hyperparameters,
- evaluation and budget settings.

Data generation currently exposes all run-dependent values as explicit CLI
arguments. See [`data_generation/`](data_generation/README.md) for the boundary
between versioned settings and local run state.
