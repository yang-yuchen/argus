# MTP whole-step graph replay: all KV lengths

Llama-7B backbone, B1, FP16, Q2, same GPU and model weights.
All27points passed correctness, fresh-input, capture and cache checks.

| KV length | Nautilus TFLOPs/s | TileOPs TFLOPs/s | FlexAttn TFLOPs/s |
|---|---:|---:|---:|
| 256 | 2.697 | 2.414 | 2.356 |
| 512 | 2.609 | 2.417 | 1.927 |
| 1024 | 2.673 | 2.432 | 1.961 |
| 2048 | 2.704 | 2.490 | 2.274 |
| 4096 | 2.873 | 2.593 | 2.532 |
| 8192 | 3.176 | 2.744 | 2.484 |
| 16384 | 3.645 | 3.255 | 3.253 |
| 32768 | 4.244 | 3.867 | 3.527 |
| 65536 | 5.166 | 4.720 | 4.699 |

Each point has5replay warmups and30synchronized wall samples.
TFLOPs/s = theoretical model FLOPs / mean replay seconds /1e12.
Original model-FLOPs convention: QKV/output projections and two MLP linears;
Q=2 for token-local work, and Q*KV-Q*(Q-1)/2 visible causal attention pairs.
The64K numerator is88851611648FLOPs. This adapts the original square-prefill
formula to the measured rectangular MTP step; it does not count prefix work.
Prefix setup, compilation, capture, LM head, draft and acceptance are excluded.
