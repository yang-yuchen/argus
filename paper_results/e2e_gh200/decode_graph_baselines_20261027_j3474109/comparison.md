# Decode whole-step CUDA Graph replay comparison

All three curves use graph replay, B1/FP16, Q1, Llama-7B backbone.
Nautilus/Flex: new measurements on gh052, job3474109. TileOPs: previous graph measurements on gh004, job3473944.

| Visible KV length | Nautilus ms | TileOPs prebound ms | FlexAttention ms |
|---|---:|---:|---:|
| 256 | 5.201 | 5.220 | 5.430 |
| 512 | 5.394 | 5.434 | 5.693 |
| 1024 | 5.475 | 5.534 | 5.881 |
| 2048 | 5.655 | 5.649 | 6.246 |
| 4096 | 5.959 | 5.953 | 6.966 |
| 8192 | 6.540 | 6.497 | 8.445 |
| 16384 | 7.631 | 7.668 | 11.349 |
| 32768 | 9.838 | 10.076 | 17.206 |

30 samples and5replay warmups perpoint. All model/fresh-input/capture checks passed.
Weights, inputs, model configuration and Torch/Transformers match at each length.
TileOPs is a cross-run/GPU reference; small differences from Nautilus are not a same-GPU controlled comparison.
