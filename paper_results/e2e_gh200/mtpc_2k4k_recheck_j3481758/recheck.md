# MTP 2K/4K graph replay recheck

Six new configurations; same GPU,30samples/5warmups, unchanged benchmark protocol.
All original data is preserved. No samples were deleted.

| KV | Backend | Old mean | Old median | New mean | New median | New max |
|---|---|---:|---:|---:|---:|---:|
| 2048 | nautilus | 13.402 | 8.447 | 8.042 | 8.042 | 8.056 |
| 2048 | tileops | 9.346 | 9.335 | 8.936 | 8.935 | 8.954 |
| 2048 | flex | 11.466 | 11.474 | 9.674 | 9.675 | 9.692 |
| 4096 | nautilus | 8.641 | 8.604 | 8.778 | 8.776 | 8.856 |
| 4096 | tileops | 9.862 | 9.859 | 9.908 | 9.907 | 9.935 |
| 4096 | flex | 25.784 | 12.448 | 11.102 | 11.099 | 11.220 |

Units: ms. The updated9-length figure replaces only these6points with the rerun;
the other21measurements are reused. Run/host/GPU provenance is retained inresults.csv.
All new correctness, changed-input, poisoned-buffer and graph capture checks passed.
