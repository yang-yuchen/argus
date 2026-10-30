# MTP whole-step graph replay: all KV lengths

Llama-7B backbone, B1, FP16, Q4, same GPU and model weights.
All27points passed correctness, fresh-input, capture and cache checks.

| KV length | Nautilus TFLOPs/s | TileOPs TFLOPs/s | FlexAttn TFLOPs/s |
|---|---:|---:|---:|
| 256 | 5.400 | 4.815 | 4.312 |
| 512 | 5.161 | 4.790 | 4.315 |
| 1024 | 5.253 | 4.856 | 4.319 |
| 2048 | 5.353 | 5.000 | 4.057 |
| 4096 | 5.664 | 5.153 | 4.960 |
| 8192 | 6.145 | 5.480 | 5.372 |
| 16384 | 7.342 | 6.490 | 6.490 |
| 32768 | 8.545 | 7.735 | 7.509 |
| 65536 | 10.214 | 9.358 | 9.372 |

Each point has5replay warmups and30synchronized wall samples.
TFLOPs/s = theoretical model FLOPs / mean replay seconds /1e12.
Original model-FLOPs convention: QKV/output projections and two MLP linears;
Q=4 for token-local work, and Q*KV-Q*(Q-1)/2 visible causal attention pairs.
The64K numerator is177701126144FLOPs. This adapts the original square-prefill
formula to the measured rectangular MTP step; it does not count prefix work.
Prefix setup, compilation, capture, LM head, draft and acceptance are excluded.
