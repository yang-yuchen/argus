# Decode whole-step CUDA Graph replay

FlexAttention · Llama-7B backbone · B1 · FP16 · Q1.
Synchronized wall latency of whole-step replay; lower is better.
Each point: 5 warmups, 30 samples. Visible KV length includes the new token.

| Visible KV length | Mean ms | Median ms | Stddev ms |
|---|---:|---:|---:|
| 256 | 5.430 | 5.428 | 0.005 |
| 512 | 5.693 | 5.693 | 0.007 |
| 1024 | 5.881 | 5.882 | 0.008 |
| 2048 | 6.246 | 6.242 | 0.020 |
| 4096 | 6.966 | 6.966 | 0.011 |
| 8192 | 8.445 | 8.444 | 0.019 |
| 16384 | 11.349 | 11.353 | 0.015 |
| 32768 | 17.206 | 17.211 | 0.021 |

All 8 cases passed model, fresh-input, graph completeness and cache checks.
All points use the same GPU, model weights, software and timing protocol.
Prefix setup, compilation, capture, tokenizer, LM head and sampling are excluded.
Fixed-position single-step decode; other backends were not rerun.
Normal compiled-forward measurements are not mixed into this replay curve.
