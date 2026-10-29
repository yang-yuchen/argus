# MTP whole-step graph replay: all KV lengths

**2K/4K已重跑**：本表保留原始全扫数据；两个长度的长尾检查与更新图见
[重跑报告](../mtpc_2k4k_recheck_j3481758/README.md)。其余21点直接复用。

Llama-7B backbone, B1, FP16, Q16, same GPU and model weights.
All27points passed correctness, fresh-input, capture and cache checks.

| KV length | Nautilus TFLOPs/s | TileOPs TFLOPs/s | FlexAttn TFLOPs/s |
|---|---:|---:|---:|
| 256 | 20.758 | 18.799 | 16.839 |
| 512 | 20.188 | 18.363 | 17.184 |
| 1024 | 20.525 | 18.917 | 16.228 |
| 2048 | 13.295 | 19.065 | 15.539 |
| 4096 | 22.608 | 19.810 | 7.577 |
| 8192 | 25.166 | 21.025 | 19.482 |
| 16384 | 28.902 | 25.103 | 20.701 |
| 32768 | 33.773 | 29.949 | 29.903 |
| 65536 | 40.723 | 36.861 | 34.939 |

Each point has5replay warmups and30synchronized wall samples.
TFLOPs/s = theoretical model FLOPs / mean replay seconds /1e12.
Original model-FLOPs convention: QKV/output projections and two MLP linears;
Q=16 for token-local work, and Q*KV-Q*(Q-1)/2 visible causal attention pairs.
The64K numerator is710754172928FLOPs. This adapts the original square-prefill
formula to the measured rectangular MTP step; it does not count prefix work.
Prefix setup, compilation, capture, LM head, draft and acceptance are excluded.
