# Nautilus Mlsys figure archive

This directory archives the Nautilus figures used by `/u/yucheny8/nautilus-mlsys27`
and the available provenance needed to find or reproduce them. The original paper
repository contains PDF and TeX files, but does not contain its plotting scripts or
CSV/JSON benchmark inputs.

## Archived figures

`figures/` contains 23 PDFs from the original 29-file paper figure directory.
The six bf16/fp8 prefill RMSE figures were excluded at the owner's request because
they were not produced by our work. `manifest.json` records every archived PDF's
SHA-256 and the excluded filenames.

The three Qwen RMSE figures are exact matches to the archived Argus artifacts and
can be regenerated offline:

```bash
cd rmse_sources
python reproduce_qwen.py
```

Their JSON values, plotting function, source log excerpts, and hashes are in
`rmse_sources/`. The causal paper filename corresponds to the `_seed1` Argus
artifact; the unsuffixed causal run is a different result.

## Performance sources

`performance_sources/` contains the available CSVs and plotting scripts for the
legacy GH200/RTX5090 performance figures, ablation, and throughput plots. Its
manifest marks exact hash matches, reusable candidates, and unresolved historical
mappings. It does not claim that a candidate CSV/script reproduces a PDF until that
has been verified.

The five canonical figures have a complete CPU-only reproduction package in
`../prefill_final_repro/`:

| Figure | Data | Script |
|---|---|---|
| GH200 FP16 compiler | `data/datasheet_new.csv` | `scripts/plot_from_csv.py` |
| GH200 FP8 compiler | `data/datasheet_fp8.csv` | `scripts/plot_from_csv.py` |
| RTX5090 FP16 compiler | `data/5090_latency_us_new.csv` | `scripts/plot_from_csv_5090.py` |
| GH200 throughput | `data/e2e_throughput.csv` | `scripts/csv_figures.py` |
| RTX5090 throughput | `data/throughput_5090new.csv` | `scripts/csv_figures_5090.py` |

`source/mlsys27/` preserves the paper's TeX, bibliography, style files and
figure presentation sources. It is reference material rather than a benchmark
reproduction source.

The newer operator results and plots are kept in their existing
`paper_results/operator_*` directories, including the FP8 full sweep and the two
best-official FP8 PDFs added in the current branch.
