# Nautilus paper results (collected 2026-10-29)

All prefill / decode / MTPC (causal MTP) / sparse results, at operator level and end-to-end (e2e).
This covers the earlier final prefill results and the Oct 25–29 runs.
Each run directory keeps its own README, plot-input CSVs, summaries and per-case JSON.
Raw nsys reports, sqlite exports, CUDA-graph traces, tuning databases and kernel binaries are not committed (~14 GB); they stay on the cluster under `/u/yucheny8/argus/profiles_*`.

Conventions:
- Operator-level plots show `Nautilus latency / method latency`: Nautilus = 1, and higher means the baseline is faster.
- Operator latency is the sum of GPU kernel durations per cold-L2 NVTX repeat, measured with nsys (50 repeats unless noted).
- "TileLang" in the decode/MTPC plots is the official TileOPs kernel (`GQADecodeBs1Kernel`, `GQADensePersistentWSKernel`).
- Speedups below are geometric means of `baseline / Nautilus` latency (>1 means Nautilus is faster).

## Final figures (`figures/`)

| Figure | Data | Plot script |
|---|---|---|
| `prefill/gh200-fp16-tc1.pdf`, `gh200-fp8-tc1.pdf`, `rtx5090-fp16-tc1.pdf` | `prefill_final_repro/data/*.csv` | `prefill_final_repro/scripts/plot_from_csv*.py` |
| `prefill/throughput_plot.pdf`, `throughput_plot_5090.pdf` (prefill e2e) | `prefill_final_repro/data/e2e_throughput.csv`, `throughput_5090new.csv` | `prefill_final_repro/scripts/csv_figures*.py` |
| `operator_gh200/gh200-fp16-tc1-decode-b7749f3.pdf` | `operator_gh200/decode_mtpc16_plot_b7749f3_20261027/decode_plot_input.csv` | `plot_from_csv.py` |
| `operator_gh200/gh200-fp16-tc1-mtpc-Q{2,4,8,16,32,64}-full-kv-optimized-20261029.pdf` | `operator_gh200/mtpc_optimize_20261028/plots/plot_input_Q*.csv` | `plot_from_csv.py` (see `mtpc_full_kv_20261028/make_plots.py`) |
| `operator_gh200/gh200-bf16-tc1-sparse-d35a206-shared-kv-tilelang.pdf` (final, Nautilus vs TileLang) | `operator_gh200/sparse_shared_kv_20261027/plot_input_tilelang.csv` | `plot_from_csv.py` |
| `operator_gh200/gh200-bf16-tc1-sparse-d35a206-shared-kv.pdf` (also has Flex/Helion BlockMask) | `operator_gh200/sparse_shared_kv_20261027/plot_input.csv` | `plot_from_csv.py` |
| `operator_gh200/gh200-fp16-tc1-attention-transition-20261029.pdf` (decode → MTP → prefill) | `operator_gh200/attention_transition_20261029/plot_input.csv` | `scripts/plot_attention_transition.py` |
| `operator_gh200/gh200-fp8-tc1-{decode8,mtpc8}-4e06b84.pdf` (FP8) | `operator_gh200/fp8_compilers_20261029/sweep_4e06b84_3484820/plot_input_*.csv` | `fp8_compilers_20261029/make_plots_fp8_sweep.py` |
| `operator_gh200/gh200-fp8-tc1-{decode8,mtpc8}-best-official-4e06b84.pdf` (FP8 best measured official route) | `operator_gh200/fp8_compilers_20261029/plot_input_*_best_official.csv` | `plot_from_csv.py` |
| `nautilus_mlsys27/` (historical figure archive and source inventory) | `nautilus_mlsys27/manifest.json` | `nautilus_mlsys27/README.md` |
| `operator_rtx5090/rtx5090-20261029-{decode,causal-mtp}.pdf` | `operator_rtx5090/rtx5090_decode_mtpc16_results.md` | `operator_rtx5090/make_plots_20261029.py` |
| `e2e_gh200/decode_e2e_graph_throughput.pdf` | `e2e_gh200/decode_graph_baselines_20261027_j3474109/decode_e2e_graph_throughput_paper_style.csv` | `scripts/plot_decode_e2e_paper_style.py` |
| `e2e_gh200/mtpc16_e2e_graph_throughput.pdf` | `e2e_gh200/mtpc_2k4k_recheck_j3481758/throughput_plot_input.csv` | `scripts/plot_mtpc_throughput_paper_style.py`, `mtpc_2k4k_recheck_j3481758/summarize_recheck.py` |

## Summary

### Prefill: earlier final version (`prefill_final_repro/`)
This is the reproducible figure package: `python reproduce.py` redraws all 5 PDFs and checks them pixel-for-pixel against `reference/`.
GH200 FP16 operator-level geomean speedups, from `datasheet_new.csv`, L = 256 … 32768:

| Operator | FlexAttn | Helion | OpenAI Triton | TAWA | TVM | TileLang | Tri-Dao Triton |
|---|---:|---:|---:|---:|---:|---:|---:|
| Global | 1.37 | 1.68 | 1.35 | 1.06 | 7.41 | 1.11 | 1.87 |
| Causal | 1.56 | 2.75 | 1.79 | 1.06 | 9.93 | 1.14 | 1.89 |
| GQA | 1.50 | 3.03 | 2.24 | 1.02 | 11.15 | 1.15 | 2.33 |

Prefill e2e average speedups (throughput, `figures/throughput_summary*.csv`):
- GH200: FlashAttn-2 1.22×, FlexAttn 1.13×, SDPA 1.23×, Tawa 1.16×.
- RTX 5090: FlashAttn-2 1.26×, FlexAttn 1.16×, SDPA 1.42×, TileLang 1.01×.

### Prefill: latest re-measurement (Oct 25–26)
- `operator_gh200/nautilus_sweep_20261026_merged/`: 44 paper kernels, Nautilus only. Covers global/causal prefill L256–32K, decode, MTP and sparse.
- `operator_gh200/prefill_decode_compare_20261026_validated/`: causal prefill vs baselines in the latest environment.
  - L256: Nautilus beats the fastest baseline by 1.10×.
  - L512–32K: cuDNN SDPA is faster than these paper kernels; its latency is 0.79–0.90× of Nautilus.
  - The historical final prefill data above was not replaced.
- `operator_gh200/prefill_official_helion_20261026/`: official Helion AOT prefill.
  - Measured points are L256 (11.74 µs) and L512 (34.46 µs).
  - L≥1024 exceeds GH200 shared memory.
- `operator_gh200/prefill_S512_*`: single-point S=512 check against the old figure (13.17 µs vs the old 13.60 µs).

### Decode, operator level (GH200, FP16, B1/H32/Q1, kernels from `b7749f3`)
Geomean over KV 256–32K: FlexAttn **2.01×**, Helion **1.48×**, TileLang (TileOPs) **1.06×**. Range against TileLang is 1.00–1.27×.
Data: `decode_mtpc16_plot_b7749f3_20261027/`, `nautilus_decode_update_b7749f3_20261027/`, `decode_mtpc16_official_tileops_20261026/`, `decode_official_recheck_20261026/`, `decode_mtpc16_tilelang_autotuned_20261026/`.

### MTPC (causal MTP, lower-right mask), operator level (GH200, FP16, H4)
- Shapes: Q ∈ {2,4,8,16,32,64} × KV ∈ {256 … 65536} = 54 shapes × 3 methods, all passing correctness.
- Four slow points (Q4/L32K, Q32/L32K, Q64/L256, Q64/L32K) were hand-tuned and re-measured (`mtpc_optimize_20261028/RESULTS.md`).
- After that, Nautilus is never slower than a baseline.
- Geomean: FlexAttn **1.86×** (1.09–3.68×), TileLang/TileOPs **1.17×** (1.003–1.60×).
- Data: `mtpc_full_kv_20261028/`, `mtpc_optimize_20261028/`, `mtpc_f8beba1_20261028/`, `mtpc16_official_flex_20261026/`.

### Sparse (DSA-style top-2048, BF16, H16, L256–32K)
- Nautilus vs official TileLang sparse MLA: geomean **1.11×** (1.06–1.15×).
- Flex/Helion (official BlockMask) are 19–26× slower at 32K, because random top-k leaves almost every block non-empty. They were dropped from the final figure.
- For L ≤ 2048 the top-2048 workload is effectively dense causal (`sparse_density_d35a206/`).
- Data: `sparse_compilers_20261027/`, `sparse_shared_kv_20261027/`, `sparse_source_audit_20261027/`.

### Cross-shape view
`attention_transition_20261029/` plots decode (Q=1) → MTP (Q=2…64) → prefill (Q=KV).
Decode and prefill use H32, while MTP uses H4 (the upstream kernel shape).

### FP8 (GH200, E4M3 Q/K/V, FP16 output)
Full-length decode8 and mtpc8 sweep, `fp8_compilers_20261029/sweep_4e06b84_3484820/`:
- Kernels: neptune-internal `4e06b84` ("more 8b kernels"), snapshot in `fp8_compilers_20261029/source_4e06b84/`.
- Run: one GH200 (gh081), job 3484820. 17 cases × 2 methods, all passed the FP8 correctness gate and the 50-repeat cold-L2 nsys measurement.
- decode8, B1/H32/Q1, KV 256–32K: Nautilus vs TileLang (official TileOPs `GQAVarlenFP8WSFwdKernel`) geomean **2.58×** (2.25–3.11×).
- mtpc8, H4/Q16 lower-right causal, KV 256–64K: vs TileLang (official TileOPs `GQADenseFP8Kernel`) geomean **5.53×** (1.71× at 256, 14.8× at 64K).
  - The selected contiguous `GQADenseFP8Kernel` has no split-KV path for this short-Q shape, so its latency grows linearly with KV. TileOPs also contains paged FP8 split kernels, but they use a paged-cache API and FP16 internal probability/PV math; they are not equivalent to this native-FP8 dense contract.
- FlexAttention is excluded: its official all-FP8 path outputs FP8, not FP16.
- Plots: `make_plots_fp8_sweep.py` (reuses `plot_from_csv.py`).

`fp8_compilers_20261029/smoke_summary.json` keeps the earlier 5-case smoke, which includes prefill8 L2048 (0.86×) and prefillc8 L2048 (1.07×).
prefill8/prefillc8/mtp8 have not been swept over all lengths.

The `best-official` PDFs use the native Dense/Varlen TileOPs result at each
length, replacing only decode8/32768 and mtpc8/65536 with measured
`GQAPagedFP8DecodeKernel` results. Paged FP8 uses FP16 internal QK/PV/probability
math, so these plots are a best-measured-official envelope rather than one
uniform kernel or precision path. The replacement policy is recorded in
`fp8_compilers_20261029/sweep_4e06b84_3484820/best_official_plot_manifest.json`.

### RTX 5090 (operator level, nsys medians)
- Newest numbers (`rtx5090_decode_mtpc16_results.md`):
  - Decode: Nautilus vs TileLang main GQA 1.17×, vs Flex 1.23×.
  - MTP-Q16: vs official Flex lower-right 1.47×.
- `rtx5090-nsys-*.pdf` / `rtx5090-*.pdf` are older plots that used the earlier, slower TileLang/Flex baselines.

### E2E (GH200, Llama-7B backbone, 32 layers, B1/FP16, whole-step CUDA Graph replay, 30 samples)
- Decode, Q1 / KV 256–32K: Nautilus 5.20 → 9.84 ms, FlexAttn 5.43 → 17.21 ms, TileOPs-prebound 5.22 → 10.08 ms.
  - Throughput geomean vs Flex **1.23×**, vs TileOPs **1.01×**.
  - Data: `decode_graph_baselines_20261027_j3474109/`; TileOPs from `decode_graph_sweep_20261027_j3473944/`.
  - Caveat: the TileOPs run was on a different node.
- MTPC, Q16 / KV 256–64K: throughput geomean vs Flex **1.23×**, vs TileOPs **1.12×**.
  - Data: `mtpc_graph_sweep_3478086/`.
  - The 2K/4K points were re-run in `mtpc_2k4k_recheck_j3481758/` because of tail-latency outliers. Final plot input: `throughput_plot_input.csv`.
- Non-graph runs (`decode_sweep_*`, `decode_bound_*`, `mtpc_native_cache_*`, …) are kept for the audit trail.
  - They are host-overhead bound and superseded by the graph-replay numbers.
- Sparse e2e: **not run**. Only a single-layer DeepSeek builder exists in `scripts/bench_paper_e2e.py`; there is no graph-replay sweep.
- Scripts: `scripts/bench_paper_e2e.py`, `scripts/run_{decode,mtp}_e2e_graph_portable.sh` (see `scripts/PORTABLE_E2E.md`).

### Early single-shape runs (Oct 25)
`single_shape_*`: the first fixed-shape pass over the paper kernels, plus the "strong baseline" search (FlashInfer, MSLK, FA3, …).
Superseded by the sweeps above, but kept for its accuracy/RMSE data.
