# Autotuned official decode versus the earlier local runner

The earlier comparison in README.md used the fixed official MHA inference example as the decode reference. The earlier local runner is a different reference: `argus/tilelang_pypi/decode.py`, imported by `scripts/bench_gh200_compare.py`.

| KV length | Earlier local (us) | Official autotuned (us) | Official slower by |
|---:|---:|---:|---:|
| 256 | 6.678 | 8.247 | 23.49% |
| 512 | 7.745 | 9.295 | 20.02% |
| 1024 | 11.194 | 12.875 | 15.01% |
| 2048 | 17.811 | 19.620 | 10.15% |
| 4096 | 30.362 | 32.477 | 6.97% |
| 8192 | 50.119 | 51.983 | 3.72% |
| 16384 | 85.290 | 87.411 | 2.49% |
| 32768 | 154.874 | 157.160 | 1.48% |

## Confirmed implementation differences

- Both implement Q=1 MHA, heads=KV heads=32, with online softmax and optional split-KV/combine. Both search the same 24 tile/stage/split configurations and already enable autotuning.
- The earlier local runner forces `TL_ENABLE_FAST_MATH=True` and `TL_DISABLE_TMA_LOWER=True`. The official example does not explicitly set these passes. Both disable warp specialization.
- The local combine kernel supplies explicit `T.annotate_layout` hints for LSE reduction arrays. The official example leaves those layouts to the compiler.
- The local runner preallocates GLSE/partial buffers and passes them as kernel inputs. The official example uses internal `T.alloc_global` buffers.
- All tested lengths are multiples of 256, so the local padding path does not affect these cases.
- Official tuning uses actual saved inputs; the earlier local autotuner did not explicitly capture those inputs. Config selection and benchmark noise may differ.

## Kernel breakdown from existing profiler traces

No new GPU run was made for this investigation. NVTX ranges were joined to CUDA runtime launches and kernel correlation IDs in the existing SQLite traces; each row below averages 50 kernel launches.

| KV length | Version | Kernel | Mean us | Registers/thread | Dynamic shared bytes | Grid |
|---:|---|---|---:|---:|---:|---|
| 256 | earlier local | `flashattn_gqa_decode_no_split_kernel` | 6.67840 | 203 | 212992 | 1x32x1 |
| 256 | official autotuned | `flashattn_gqa_decode_no_split_kernel` | 8.24704 | 204 | 212992 | 1x32x1 |
| 8192 | earlier local | `flashattn_gqa_decode_split_kernel` | 48.98496 | 240 | 81920 | 1x32x8 |
| 8192 | earlier local | `flashattn_gqa_decode_split_kernel_1` | 1.13408 | 32 | 0 | 32x1x1 |
| 8192 | official autotuned | `flashattn_gqa_decode_split_kernel` | 50.77184 | 148 | 49152 | 1x32x8 |
| 8192 | official autotuned | `flashattn_gqa_decode_split_kernel_1` | 1.21152 | 32 | 0 | 32x1x1 |
| 32768 | earlier local | `flashattn_gqa_decode_split_kernel` | 153.66208 | 240 | 81920 | 1x32x8 |
| 32768 | earlier local | `flashattn_gqa_decode_split_kernel_1` | 1.21152 | 32 | 0 | 32x1x1 |
| 32768 | official autotuned | `flashattn_gqa_decode_split_kernel` | 155.81502 | 238 | 81920 | 1x32x8 |
| 32768 | official autotuned | `flashattn_gqa_decode_split_kernel_1` | 1.34528 | 32 | 0 | 32x1x1 |

The gap is predominantly in the main attention kernel, not the combine stage. CPU allocation/launch overhead is excluded from the reported GPU kernel duration sum; it must not be used to explain this measured gap directly.

## What remains unproven

The code and traces identify different compiler settings/layouts, different compiled resources, and where the time is spent. They do not isolate the causal contribution of each difference. That requires a same-device experiment with matched configurations and one pass/layout change at a time. The earlier local measurements were on gh151, the official autotuned sweep on gh032; small long-sequence gaps can include node/run variation.

MTP is a separate comparison: the same official MHA varlen kernel improved from fixed 128x128/stages2/threads256 to autotuned 64x128/stages1/threads128, giving 1.29-1.37x throughput (about 22.5-27.1% lower latency).
