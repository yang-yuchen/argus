# Updated causal MTP: latest kernels and original compiler baselines

## Completed run

- Upstream SSH remote: `git@github.com:uiuc-arc/neptune-internal.git`, branch `nautilus-gh200`.
- Pinned latest commit: `f8beba14d7af0f49354809179d6db580837e74ee`.
- New causal-MTP schedules were added by `c1a4a60b2e6797fd9c332ae47e39e171c09206ed`.
- Slurm job `3478832`, node `gh066`, one GH200, completed `0:0` in 42m50s.
- **29 shapes × 3 methods = 87 validated measurements.** Q16 was remeasured rather than reusing cross-node measurements.

| Query count Q | Measured total KV lengths |
|---:|---|
| 2, 4, 8, 32, 64 | 1024, 4096, 16384, 65536 |
| 16 | 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 |

Every case is FP16, B1/H4/D128, Q `[1,4,Q,128]`, K/V `[1,4,L,128]`, output `[1,4,Q,128]`.
The lower-right causal predicate is `key_position <= L - Q + query_position`.
Real Q counts are used even where the upstream schedule pads Q to 16 internally.

## Kernel provenance

New Q cases in GitHub contain `tilelang/best/schedule.py`, `trace.py` and `result.txt`, but no binary export.
The original complete winner exports were available in the upstream tuning workspace:

```text
/work/nvme/beqg/mbudiu2/paper_kernels_sweep/L{L}/mtpc_Q{Q}_L{L}/tilelang_w/best/export
```

All 20 new exports were copied without modification. Each exported `schedule.py` is byte-identical
to the corresponding schedule at the pinned GitHub commit, including the four warm-started winners.
Export files also match their original tuning files. No attention kernel or schedule was rewritten,
and no Nautilus retuning or recompilation was required. The 29 exports are frozen under `kernels/`;
[source_manifest.json](source_manifest.json) records per-file hashes and exact provenance.

## Methods

- **Nautilus:** upstream winner `kernel.so`, loaded in the existing compatible Neptune TVM runtime
  through `scripts/bench_paper_sweep.py`.
- **FlexAttn:** official PyTorch `flex_attention`, lower-right `BlockMask`,
  `fullgraph=True, mode=max-autotune-no-cudagraphs`, inference mode.
  Existing runner: `scripts/bench_official_flex_mtpc.py`.
- **TileLang (plot label):** official TileOPs `GQADenseFwdOp(is_causal=True,target=BUILTIN)`,
  commit `85e7e644f1d99037e937378b8d8dc70c6568113f`, Hq=Hkv=4 gives MHA.
  Existing runner: `scripts/bench_official_tileops_dense.py`.
  This follows the user's earlier choice to label TileOPs results TileLang.
  All 29 cases select `GQADensePersistentWSKernel`. The public default dispatch/configuration
  follows the previous measurements; this run did not request a broader TileOPs autotune search.

Both compiler baselines use `/work/nvme/beqg/yucheny8/envs/argus-gh200-latest`.
Their existing scripts were extended to accept Q and the saved input file; upstream kernel algorithms
were not edited. Tests for dynamic input shapes, lower-right reference behavior and profiler-child
argument forwarding passed (17 focused checks; an earlier broader compatibility run passed 33).

An unnecessary separate varlen TileLang probe was briefly submitted as job `3479319` on gh021 and
cancelled after 1m56s when the user's TileOPs labeling choice was reapplied. It produced no completed
measurement. Its logs remain separate in `official_tilelang/`; none are used in these tables or PDFs.

## Measurement and checks

All methods load identical saved Q/K/V bytes for each case. Before timing, output is checked against
the existing chunked FP32 reference with TF32 disabled, rtol=1e-2, atol=3e-3, relative L2<1e-2.
Maximum observed relative L2 is below 0.000385. Each method performs 10 warmups, then 50 repeats with
200 MiB cold-L2 flushes. The metric is the sum of **all GPU kernel durations** within each repeat's NVTX
range, including split reductions. CPU launches, compilation/autotuning and layout preparation are excluded.

`make_plots.py` independently checks completeness, finite positive samples, sample means, input hashes,
Nautilus kernel hashes, error limits, timing metadata, one common host and per-Q length coverage before
writing plots. Existing raw results and prior figures remain intact.

The largest sample CV is 14.2%, for TileLang Q16/L256: one of 50 samples is 13.504us, while the median
is 6.688us and mean 6.816us. It is retained rather than discarded. Nautilus and Flex maximum CVs are
3.21% and 2.64%. These are sequential measurements within one GPU allocation; no interleaved drift
control was performed. This experiment establishes the performance of the recorded official configurations,
not global optimality over all configurations or official implementations.

## Artifacts

- [All latencies in microseconds](latencies_us.csv)
- [Comparison and speedups](comparison.csv)
- [87-case completion summary](measurement_summary.json)
- [Plot verification manifest](plot_manifest.json)
- Raw nsys reports, CSVs, JSONs and inputs: `nautilus/`, `flex/`, `tileops/`.

Each PDF reuses `plot_from_csv.py` with `Y = Nautilus latency / method latency`: Nautilus=1,
and **higher is faster**. The TileLang curve always uses this run's TileOPs results; no old Q16
varlen results are substituted.

| Q | PDF | Geometric mean Nautilus speedup over TileLang | Over FlexAttn |
|---:|---|---:|---:|
| 2 | [Q2](../../plots_from_csv/gh200-fp16-tc1-mtpc-Q2-f8beba1.pdf) | 1.1450x | 1.4092x |
| 4 | [Q4](../../plots_from_csv/gh200-fp16-tc1-mtpc-Q4-f8beba1.pdf) | 1.1423x | 1.4375x |
| 8 | [Q8](../../plots_from_csv/gh200-fp16-tc1-mtpc-Q8-f8beba1.pdf) | 1.1337x | 1.4823x |
| 16 | [Q16](../../plots_from_csv/gh200-fp16-tc1-mtpc-Q16-f8beba1.pdf) | 1.2370x | 1.9118x |
| 32 | [Q32](../../plots_from_csv/gh200-fp16-tc1-mtpc-Q32-f8beba1.pdf) | 1.1359x | 1.8753x |
| 64 | [Q64](../../plots_from_csv/gh200-fp16-tc1-mtpc-Q64-f8beba1.pdf) | 1.1326x | 2.0784x |

Revalidate raw results and regenerate all six PDFs with:

```bash
MPLCONFIGDIR=/tmp/argus-matplotlib-mtpc \
/work/nvme/beqg/yucheny8/envs/argus-gh200-latest/bin/python \
  profiles_paper_kernels/mtpc_f8beba1_20261028/make_plots.py
```
