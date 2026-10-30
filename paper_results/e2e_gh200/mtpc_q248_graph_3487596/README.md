# MTP Q2/Q4/Q8 whole-model CUDA Graph replay

Completed on GH200 host `gh151`, GPU UUID
`5fec3f12-dcb5-1e2c-ff5b-713d0f6aa983`, job `3487596`.

Each query length has KV lengths 256, 512, 1024, 2048, 4096, 8192, 16384,
32768 and 65536. Every shape compares Nautilus, official TileOPs
`GQADensePersistentWSKernel` and FlexAttention. Each point has 5 graph replay
warmups and 30 synchronized wall samples. The full graph includes new KV writes,
projections, attention, MLP, norm and residual; setup, compilation, LM head and
sampling are outside the timed step.

The six source files in `Q2`, `Q4` and `Q8` are the JSON/CSV/Markdown/log result
records. Large Chrome graph traces stay under the corresponding `profiles_paper_e2e`
directory. The plot-input CSVs are `mtp_q{2,4,8}_plot_input.csv`; paper-style
figures are in `paper_results/figures/e2e_gh200/mtpc_q{2,4,8}_e2e_graph_throughput.*`.

Reproduce the summaries/throughput inputs from the source results with:

```bash
python scripts/summarize_mtpc_graph_sweep.py DIRECTORY --query-length 2
python scripts/summarize_mtpc_graph_sweep.py DIRECTORY --query-length 4
python scripts/summarize_mtpc_graph_sweep.py DIRECTORY --query-length 8
```
