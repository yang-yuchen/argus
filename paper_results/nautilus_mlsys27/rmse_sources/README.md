# RMSE figure source archive

This directory records the nine RMSE PDFs used by `nautilus-mlsys27` at the
`cced96b` Argus snapshot. Hashes are in `SHA256SUMS` and `rmse_manifest.json`.
No model weights or credentials are included.

## Status

- **exact-match:** The Qwen global and GQA PDFs are byte-identical to the
  corresponding Argus outputs in `argus_artifacts/`. The paper's causal PDF is
  byte-identical to the Argus `...causal_seed1.pdf` output; the filename in the
  paper omits `_seed1`.
- **reconstructable:** All three Qwen figures have 1,000 per-sample values in
  archived JSON. Install `requirements.txt`, then run `python reproduce_qwen.py`;
  it loads the archived
  `scripts/error.py` plot function and writes three PDFs under `reconstructed/`.
  Reconstructed PDFs can differ in PDF metadata while preserving plotted data.
`run_log_excerpts.txt` and `extract_argus_error_summary.py` preserve the
available Qwen provenance. The unrelated bf16/fp8 prefill RMSE figures were
excluded from this archive at the owner's request.
