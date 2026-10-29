# Portable decode / MTP graph e2e

Copy these scripts into the `scripts/` directory of an Argus checkout. They do not contain
machine-specific `/u/yucheny8` or `/work/...` paths. The checkout still needs the matching
`paper_kernels/`, `.tileops_work/` and optional `.paper_e2e_hf/` directories.

Set the environment once:

```bash
export ARGUS_ENV_PREFIX=/path/to/argus-gh200-latest
export CONDA_BASE=/path/to/miniconda3       # optional if `conda` is already on PATH
```

For Nautilus, also set its runtime paths:

```bash
export NAUTILUS_RUNTIME_ROOT=/path/to/argus-paper-kernels/runtime_py313
export NAUTILUS_TVM_LIBRARY=/path/to/neptune/build/temp.linux-aarch64-cpython-311
```

Run one backend per output directory:

```bash
bash scripts/run_decode_e2e_graph_portable.sh OUT/decode_tileops tileops_bound
bash scripts/run_decode_e2e_graph_portable.sh OUT/decode_flex flex
bash scripts/run_decode_e2e_graph_portable.sh OUT/decode_nautilus nautilus

bash scripts/run_mtp_e2e_graph_portable.sh OUT/mtp_tileops tileops
bash scripts/run_mtp_e2e_graph_portable.sh OUT/mtp_flex flex
bash scripts/run_mtp_e2e_graph_portable.sh OUT/mtp_nautilus nautilus
```

Decode runs 8 KV lengths (256 through 32768); MTP runs 9 KV lengths (256 through 65536).
Each point uses 5 warmups and 30 replay samples by default. Override with `WARMUP` and
`REPEAT`. The scripts emit the per-length JSON and graph trace files. Decode summaries use
`scripts/summarize_decode_e2e_graph.py`; the completed MTP sweep uses
`scripts/summarize_mtpc_graph_sweep.py`.

These are portable within an Argus checkout plus its runtime dependencies; they are not
standalone scripts that can run without the package, exported kernels or a GHx4 GPU.
