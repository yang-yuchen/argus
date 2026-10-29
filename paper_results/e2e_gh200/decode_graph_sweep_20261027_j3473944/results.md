# Decode whole-step CUDA Graph replay

TileOPs prebound · Llama-7B backbone · B1 · FP16 · Q1.
Synchronized wall latency of whole-step replay; lower is better.
Each point: 5 warmups, 30 samples. Visible KV length includes the new token.

| Visible KV length | Mean ms | Median ms | Stddev ms |
|---|---:|---:|---:|
| 256 | 5.220 | 5.220 | 0.004 |
| 512 | 5.434 | 5.434 | 0.005 |
| 1024 | 5.534 | 5.535 | 0.011 |
| 2048 | 5.649 | 5.649 | 0.006 |
| 4096 | 5.953 | 5.954 | 0.005 |
| 8192 | 6.497 | 6.495 | 0.006 |
| 16384 | 7.668 | 7.666 | 0.007 |
| 32768 | 10.076 | 10.089 | 0.038 |

All 8 cases passed model, fresh-input, graph completeness and cache checks.
All points use the same GPU, model weights, software and timing protocol.
Prefix setup, compilation, capture, tokenizer, LM head and sampling are excluded.
Fixed-position single-step decode; other backends were not rerun.
Normal compiled-forward measurements are not mixed into this replay curve.
