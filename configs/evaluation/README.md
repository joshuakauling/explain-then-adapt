# Evaluation Configuration

`evaluation.yaml` fixes the parser limits and the token-throughput assumptions
used for the thesis results.

The cost model is:

```text
C_total = (T_prefill + 3 * T_train) / r_prefill + T_gen / r_decode
```

`T_prefill` includes Reasoning Model and Prediction Model prompt tokens,
`T_train` is the sum of sequence tokens processed by TTT, and `T_gen` includes
Reasoning Model traces and Prediction Model outputs. The checked-in A100 rates
produce the seconds-equivalent values reported in the thesis. They should be
changed, and the resulting report labelled accordingly, for another hardware
profile.
