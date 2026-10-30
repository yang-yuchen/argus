# Nautilus Mlsys27 historical performance sources

This directory preserves the CPU inputs and plotting code used for the GH200/RTX5090 performance figures and `ablation_study.pdf`. The target paper commit is recorded in `performance_manifest.json`; `figures_sha256/historical_pdfs.sha256` records the original PDFs in `/u/yucheny8/nautilus-mlsys27` and `figures/old`.

The five fully verified figures (`gh200-fp16-tc1`, `gh200-fp8-tc1`, `rtx5090-fp16-tc1`, and both throughput plots) use the existing [`prefill_final_repro`](../../prefill_final_repro) package; its scripts and data are intentionally not duplicated. The remaining GH200 baseline/tc variants and RTX5090 historical variants retain candidate source CSVs and the original script variants. Their PDF hashes are recorded, but no equivalence is claimed where the exact historical input/layout cannot be proven.

## CPU redraw examples

```bash
mkdir -p /tmp/nautilus-ablation-redraw
cp data/ablation.csv scripts/ablation_figure.py /tmp/nautilus-ablation-redraw/
(cd /tmp/nautilus-ablation-redraw && python3 ablation_figure.py)
python3 ../../prefill_final_repro/reproduce.py        # verified five-figure package
```

For the standalone scripts, pass paths with their original CLI options (`--input-fp16`, `--input-fp8`, `--output-dir`; see each script). No CUDA benchmark is run here and missing measurements remain missing.

Source provenance: Argus working tree at commit `cced96b...`; throughput scripts/data copied from `/u/yucheny8/neptune-internal`. Historical PDF identity is based on SHA256; Poppler/render checks are documented by `prefill_final_repro`.

The stored scripts are unchanged historical sources. The ablation command redraws the available source variant; exact identity with the paper PDF is not established. GH200 `baseline1`/`baseline1new` and RTX5090 `baseline1` match existing local PDFs byte for byte, which proves artifact identity only, not a source redraw. Old RTX5090 `baseline1`/`tc1` match `plots_from_csv`; old RTX5090 throughput matches the original neptune-internal `throughput_plot_5090.pdf` byte for byte.
