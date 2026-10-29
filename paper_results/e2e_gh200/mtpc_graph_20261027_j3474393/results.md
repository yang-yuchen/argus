# MTP whole-step CUDA Graph replay

Llama-7B backbone · B1 · FP16 · Q16 · visible KV65536 · 32 layers.
Same GPU, model weights, inputs and timing protocol for all three implementations.
Each point: 5 replay warmups, 30 synchronized wall samples. Lower is better.

| Implementation | Mean ms | Median ms | Stddev ms | Latency / Nautilus |
|---|---:|---:|---:|---:|
| nautilus | 17.089 | 17.088 | 0.009 | 1.000× |
| tileops | 19.011 | 19.009 | 0.030 | 1.112× |
| flex | 19.311 | 19.309 | 0.010 | 1.130× |

All three cases passed model, fresh-input, graph completeness and cache checks.
The full step includes new-token KV writes, all 256 attention groups and merges.
Prefix setup, compilation, capture, tokenizer, LM head and sampling are excluded.
Fixed-position candidate verification; draft generation and acceptance are excluded.
Normal compiled-forward results are not mixed into this replay comparison.
