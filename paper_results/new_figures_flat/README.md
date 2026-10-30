# Flat paper-style figure review set

There are18 PNGs in one directory:12 operator previews and6 E2E experiments.
IDs3,13,15,16 were removed as requested. Figure5 has the corrected transition
layout. Other retained IDs are preserved.

Every E2E preview now uses the original paper's grouped throughput bars, method
colors and legend, Throughput(TFLOPs/s) vertical axis and power-of-two KV labels.
The original plot_subplot function is reused directly. Eager and CUDA Graph
measurements remain different experiments, explicitly recorded in index.csv.
No benchmark was rerun and no measurement was substituted between experiments.

E2E review IDs:
- 1: decode CUDA Graph, all three methods; replaces latency/single-method views20–24.
- 2: final MTP CUDA Graph sweep with the2K/4K rechecks.
- 19: eager decode comparison including the prebound TileLang route.
- 25: earlier eager decode comparison.
- 26: earlier single64K MTP CUDA Graph experiment; merges original views26/27.
- 28: original MTP CUDA Graph sweep before the2K/4K rechecks.

The standard E2E PDFs, exact plot-input CSVs and manifest are under
paper_results/e2e_gh200/standard_review. index.csv records the underlying result
CSV for every E2E preview and identifies which old presentation IDs it replaces.
