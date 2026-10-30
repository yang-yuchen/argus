# MTP whole-step graph replay: all KV lengths

Llama-7B backbone, B1, FP16, Q8, same GPU and model weights.
All27points passed correctness, fresh-input, capture and cache checks.

| KV length | Nautilus TFLOPs/s | TileOPs TFLOPs/s | FlexAttn TFLOPs/s |
|---|---:|---:|---:|
| 256 | 11.118 | 9.879 | 9.394 |
| 512 | 10.651 | 9.821 | 8.379 |
| 1024 | 10.746 | 9.990 | 8.424 |
| 2048 | 11.175 | 10.225 | 8.440 |
| 4096 | 11.721 | 10.506 | 9.632 |
| 8192 | 12.922 | 11.154 | 10.504 |
| 16384 | 15.088 | 13.279 | 12.858 |
| 32768 | 16.797 | 15.773 | 15.353 |
| 65536 | 20.865 | 19.036 | 18.732 |

Each point has5replay warmups and30synchronized wall samples.
TFLOPs/s = theoretical model FLOPs / mean replay seconds /1e12.
Original model-FLOPs convention: QKV/output projections and two MLP linears;
Q=8 for token-local work, and Q*KV-Q*(Q-1)/2 visible causal attention pairs.
The64K numerator is355393863680FLOPs. This adapts the original square-prefill
formula to the measured rectangular MTP step; it does not count prefix work.
Prefix setup, compilation, capture, LM head, draft and acceptance are excluded.
