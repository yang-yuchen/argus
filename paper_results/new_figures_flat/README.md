# Standard operator/e2e figure previews

All18 PNGs are in this directory, with no subdirectories:

- 14 GH200 operator figures (relative-performance curves).
- 2 current RTX5090 operator figures (relative-performance curves).
- 2 GH200 E2E figures (paper-style throughput bars).

The RTX5090 previews use the current PDFs in paper_results/operator_rtx5090,
including the changes from commit2b4a864. Every preview is rendered directly from
its recorded source PDF at144dpi. index.csv records each source and its SHA256.
The transition figure keeps the averaged-across-KV curves requested for the paper.

The review set uses the final paper-style counterparts for intermediate latency,
single-method and diagnostic presentations, and removes duplicate presentations.
The original plots and experiment data remain in their experiment directories.
