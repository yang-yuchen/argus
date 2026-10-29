# RTX 5090 result plots

Generated from the two original JSON files using the existing
`plot_from_csv_5090.py` load, normalization and plotting functions.
No benchmark was rerun and the source JSON files were not modified.

- [Decode PDF](rtx5090-decode.pdf): Nautilus, FlexAttn, TileLang.
- [Causal MTP Q16 PDF](rtx5090-causal-mtp.pdf): Nautilus, FlexAttn.
- Each plot covers sequence lengths 256 through 32768, sorted numerically.
- Input CSV values are the saved **median_ms**, from 5 rounds per successful result.
- Y = Nautilus median latency / method median latency. Nautilus=1; higher means faster.
- `tilelang-<variant>` is labeled Nautilus (the generated exports in the supplied
  `nautilus/local/rtx5090/.../paper-tilelang` result paths).
  `tilelang-official-*` is the separate TileLang baseline. `flex-5090` is FlexAttn.
- Following the compiler-baseline scope, `sdpa-5090` is excluded; the JSON does
  not identify its selected SDPA backend. FlashInfer decode is unsupported.
- All eight official TileLang MTP measurements failed autotuning; they are
  omitted, not plotted as zeros or substituted with another kernel.
- The saved JSON correctness/status flags and positive finite latencies were
  checked. This plotting task does not independently revalidate the kernels,
  timing protocol or baseline provenance.

Exact source hashes, method mapping outcomes and omissions are recorded in
[rtx5090_plot_manifest.json](rtx5090_plot_manifest.json).
