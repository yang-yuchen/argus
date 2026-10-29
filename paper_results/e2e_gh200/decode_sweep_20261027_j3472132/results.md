# Decode model forward latency

Llama-7B backbone, B1, FP16, Q1; synchronized wall latency in ms.
Persistent native KV cache; no full KV staging. Lower latency is better.

| KV length | Nautilus ms | TileOPs ms | FlexAttention ms | TileOPs / Nautilus | Flex / Nautilus |
|---|---:|---:|---:|---:|---:|
| 256 | 25.656 | 28.130 | 27.216 | 1.096× | 1.061× |
| 512 | 23.138 | 26.698 | 28.100 | 1.154× | 1.214× |
| 1024 | 23.105 | 31.227 | 28.032 | 1.352× | 1.213× |
| 2048 | 23.335 | 29.082 | 26.771 | 1.246× | 1.147× |
| 4096 | 23.518 | 31.266 | 27.312 | 1.329× | 1.161× |
| 8192 | 23.856 | 28.338 | 27.612 | 1.188× | 1.157× |
| 16384 | 23.819 | 29.832 | 27.520 | 1.252× | 1.155× |
| 32768 | 23.658 | 31.342 | 27.721 | 1.325× | 1.172× |

All 24 cases passed validation; 20 samples per case.
Ratios are each method's mean latency divided by Nautilus at the same KV length.
Initialization, prefix preparation, tokenizer, LM head and sampling are excluded.
