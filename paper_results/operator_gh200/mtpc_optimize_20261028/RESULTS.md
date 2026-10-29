# Completed MTPC slow-point optimization

All four optimized exports passed the original accuracy gates and complete50-repeat cold-L2 Nsight measurements. Saved original and baseline GH200 measurements were reused across cluster nodes, as requested.

| Q / KV | Original Nautilus (us) | Optimized Nautilus (us) | TileLang (us) | Throughput gain over TileLang |
|---|---:|---:|---:|---:|
| 4 / 32768 | 33.35 | 30.94 | 31.54 | 1.9% |
| 32 / 32768 | 35.26 | 30.63 | 32.35 | 5.6% |
| 64 / 256 | 9.23 | 5.58 | 6.84 | 22.5% |
| 64 / 32768 | 38.17 | 31.18 | 33.34 | 6.9% |

With these four replacements, none of the54 measured shapes is slower than either saved TileLang or FlexAttn baseline. The smallest TileLang margin among these four is1.94%, so small margins should not be interpreted as a large performance separation.

Inputs and export hashes were verified; each final measurement has50 samples,10 warmups and200MiB L2 flushes. Candidate validation additionally used original and seed17 inputs, NaN/one output poisoning, unchanged rtol1e-2/atol3e-3 and relativeL2<1e-2.

Only upstream schedule/trace parameters were changed. Q4 reused the published same-Q trace with KV sampling adjustments; the other three used schedule_split_kv_flash. Actual automatic records: Q4=96, others=64 each; the256-trial ceiling was not completed and no global optimum is claimed.

Original full-sweep data and figures were preserved. Updated PDFs are listed below. Optimized comparison: comparison.csv. Full measurement metadata: measurement/manual/*.json.

## Updated PDFs

- [Q2](../../plots_from_csv/gh200-fp16-tc1-mtpc-Q2-full-kv-optimized-20261029.pdf)
- [Q4](../../plots_from_csv/gh200-fp16-tc1-mtpc-Q4-full-kv-optimized-20261029.pdf)
- [Q8](../../plots_from_csv/gh200-fp16-tc1-mtpc-Q8-full-kv-optimized-20261029.pdf)
- [Q16](../../plots_from_csv/gh200-fp16-tc1-mtpc-Q16-full-kv-optimized-20261029.pdf)
- [Q32](../../plots_from_csv/gh200-fp16-tc1-mtpc-Q32-full-kv-optimized-20261029.pdf)
- [Q64](../../plots_from_csv/gh200-fp16-tc1-mtpc-Q64-full-kv-optimized-20261029.pdf)

Plot inputs and the merged54-shape comparison are in plots/.
