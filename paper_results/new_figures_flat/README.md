# Flat paper-style figure review set

There are 14 PNGs in one directory: 12 operator previews and 2 E2E figures.
IDs 3, 13, 15, 16, 19, 25, 26, 28 were retired; retained IDs are stable.
Figure 5 has the repaired transition layout.

E2E titles contain model, dtype, batch/query size, operator and GPU.
Both E2E figures reuse the original paper grouped throughput bars and colors.
The index retains the execution protocol and exact data sources.

- 1: decode CUDA Graph comparison.
- 2: corrected full MTP CUDA Graph sweep, including all six rerun 2K/4K measurements.

26 was the earlier 64K-only MTP run. 28 was the full sweep before its 2K/4K
outliers were rerun. Figure 2 uses the corrected full sweep.
Raw experiment files remain in their original folders.

See [EXPERIMENTS.md](EXPERIMENTS.md) for retained and omitted experiments, and
`../e2e_gh200/standard_review/manifest.json` for E2E provenance.
