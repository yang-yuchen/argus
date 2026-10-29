# Nautilus kernel update b7749f3

Upstream: `uiuc-arc/neptune-internal`, branch `nautilus-gh200`, commit `b7749f37d2aa96b7124d2a97c2bfec5be1011346` (2026-10-27).

Only decode L4096, L8192, L16384 and L32768 binaries changed. All four were remeasured using `scripts/bench_paper_sweep.py`, original saved input bytes, independent FP32 correctness checks, 10 warmups, 200 MiB L2 flush and 50 repeats. Timing sums GPU kernel durations per repeat. Slurm job `3468181` ran on GH200 `gh011`.

The other four decode and all nine causal MTP binaries match the previous measured SHA256 exactly. Their correctness-validated measurements are reused. The MTP curve therefore remains numerically unchanged.

TileLang uses the measured official TileOPs kernels, under the legend `TileLang`. FlexAttention and Helion values are reused unchanged. Plotting reuses `plot_from_csv.py`; normalization is Nautilus latency divided by each method latency, so higher means faster.

| Updated decode KV length | Previous mean us | New mean us | Previous/new |
|---:|---:|---:|---:|
| 4096 | 30.18752 | 29.66592 | 1.0176x |
| 8192 | 50.39680 | 49.61408 | 1.0158x |
| 16384 | 89.34400 | 85.38304 | 1.0464x |
| 32768 | 163.55200 | 154.32768 | 1.0598x |

Updated padded decode exports are nested under `tilelang/tilelang_auto/best/export`. Their ABI remains `main(Q,K,V,O)` with Q/output `[1,32,1,128]`. Padding/cropping occurs inside the official exported kernel.

The previous source tree is preserved at `paper_kernels.before_b7749f3_20261027/`; baseline records and previous plots remain available.

The baseline and earlier Nautilus measurements come from earlier runs/nodes. The figure is a cross-run comparison; small differences include possible node/clock variation.

## Reproduction

```bash
/work/nvme/beqg/yucheny8/envs/argus-gh200-latest/bin/python plot_from_csv.py --input-fp16 profiles_paper_kernels/plot_update_b7749f3_20261027/decode_plot_input.csv --operators DC_CAUSAL --suffix=-fp16-tc1-decode-b7749f3 --output-dir plots_from_csv
/work/nvme/beqg/yucheny8/envs/argus-gh200-latest/bin/python plot_from_csv.py --input-fp16 profiles_paper_kernels/plot_update_b7749f3_20261027/mtp_plot_input.csv --operators MTP_CAUSAL --suffix=-fp16-tc1-mtpc-b7749f3 --output-dir plots_from_csv
```
