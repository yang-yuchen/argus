# Hopper baselines: measured candidates passing the existing accuracy gates

RMSE uses scripts.error.compute_rmse and FP64 reference. Times are cold-L2 GPU kernel sums.
These H100/SM90-compatible candidates were measured on GH200, not a physical H100.

| Variant | Backend | Candidate | Mean us | RMSE | Nautilus us | Baseline/Nautilus |
|---|---|---|---:|---:|---:|---:|
| decode | PyTorch Flash | torch_flash | 10.972 | 1.991e-05 | 7.072 | 1.551x |
| decode | cuDNN | torch_cudnn | 9.170 | 1.999e-05 | 7.072 | 1.297x |
| decode | FlashAttention-2 | flashattention2 | 10.638 | 1.990e-05 | 7.072 | 1.504x |
| decode | FlashInfer | flashinfer_simt | 8.472 | 2.127e-05 | 7.072 | 1.198x |
| decode | TileLang | tilelang_decode_n128_h64_s4 | 9.390 | 5.802e-05 | 7.072 | 1.328x |
| decode | Triton MSLK | triton_mslk_s8 | 8.603 | 1.919e-05 | 7.072 | 1.216x |
| decode | Triton vLLM | triton_vllm_s4 | 11.480 | 7.669e-05 | 7.072 | 1.623x |
| mtp | PyTorch Flash | torch_flash | 71.007 | 1.781e-06 | 52.035 | 1.365x |
| mtp | cuDNN | torch_cudnn | 964.980 | 1.931e-06 | 52.035 | 18.545x |
| mtp | FlashAttention-2 | flashattention2 | 74.490 | 1.782e-06 | 52.035 | 1.432x |
| mtp | FlashInfer | flashinfer_ragged_fa2 | 57.658 | 2.565e-06 | 52.035 | 1.108x |
| mtp | TileLang | tilelang_sm90_mtp_m64_n128_s16 | 77.133 | 1.012e-05 | 52.035 | 1.482x |
| mtp | Triton MSLK | triton_mslk_s128 | 55.438 | 1.746e-06 | 52.035 | 1.065x |
| mtp | Triton vLLM | triton_vllm_s64 | 432.187 | 6.102e-06 | 52.035 | 8.306x |
| sparse | TileLang | tilelang_i64_s1_t64 | 868.326 | 2.013e-04 | 764.001 | 1.137x |
| sparse | FlashInfer preplanned | flashinfer_mla_fa3_preplanned | 384.017 | 2.010e-04 | 764.001 | 0.503x |
| sparse | FlashInfer | flashinfer_mla_fa3_timed_compaction | 418.335 | 2.010e-04 | 764.001 | 0.548x |
| sparse_split | TileLang | tilelang_i64_s1_t64 | 893.100 | 2.013e-04 | 1077.068 | 0.829x |
| sparse_split | FlashInfer preplanned | flashinfer_mla_fa3_preplanned | 382.432 | 2.010e-04 | 1077.068 | 0.355x |
| sparse_split | FlashInfer | flashinfer_mla_fa3_timed_compaction | 417.861 | 2.010e-04 | 1077.068 | 0.388x |

Ratios use the same-GPU Nautilus measurement from each variant's cohort.
Preplanned FlashInfer excludes CSR compaction and is reported separately.
All-cases equivalence and any Nautilus failed probes are recorded in best_by_backend.csv.

## Accuracy qualifications

Passing the existing tolerance does not mean identical RMSE.
rmse_matched_best.csv additionally requires benchmark RMSE <= Nautilus RMSE.
- Nautilus sparse_split failed: padding_duplicates. Its benchmark timing is valid for the nonempty-row benchmark input only; full-domain accuracy equivalence is not established.
