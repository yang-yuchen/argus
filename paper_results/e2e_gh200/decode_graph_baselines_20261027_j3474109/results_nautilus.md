# Decode whole-step CUDA Graph replay

Nautilus · Llama-7B backbone · B1 · FP16 · Q1.
Synchronized wall latency of whole-step replay; lower is better.
Each point: 5 warmups, 30 samples. Visible KV length includes the new token.

| Visible KV length | Mean ms | Median ms | Stddev ms |
|---|---:|---:|---:|
| 256 | 5.201 | 5.200 | 0.006 |
| 512 | 5.394 | 5.395 | 0.008 |
| 1024 | 5.475 | 5.475 | 0.006 |
| 2048 | 5.655 | 5.653 | 0.012 |
| 4096 | 5.959 | 5.959 | 0.007 |
| 8192 | 6.540 | 6.539 | 0.010 |
| 16384 | 7.631 | 7.626 | 0.013 |
| 32768 | 9.838 | 9.838 | 0.007 |

All 8 cases passed model, fresh-input, graph completeness and cache checks.
All points use the same GPU, model weights, software and timing protocol.
Prefix setup, compilation, capture, tokenizer, LM head and sampling are excluded.
Fixed-position single-step decode; other backends were not rerun.
Normal compiled-forward measurements are not mixed into this replay curve.
