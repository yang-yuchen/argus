# Complete causal-MTP KV-length sweep (in progress)

Target coverage: Q in **2,4,8,16,32,64**, each at KV length
**256,512,1024,2048,4096,8192,16384,32768,65536**.
That is **54 shapes × 3 methods = 162 measurements**.

## Existing data

The 29 previously published shapes and their 87 measured Nautilus/Flex/TileOPs
results are linked from `../mtpc_f8beba1_20261028/`, without rewriting those
records. Their kernel binaries were checked against the latest upstream commit
`5c7555b4e51b4b0c966a9b4ef84d8ed764fa81c7`: all 29 match byte-for-byte.
See `prior_kernel_latest_checks.json`.

## Missing 25 shapes

The upstream paper directory does not provide the other 25 shapes. They will
be generated locally from the **unchanged upstream compute and tuning code**:

```text
Q={2,4,8,32,64} × KV={256,512,2048,8192,32768}
```

- Nautilus source commit: `5c7555b4e51b4b0c966a9b4ef84d8ed764fa81c7`.
- TileLang submodule: `04b88a92305ddd4ce8c463d48656e3ff1b910edc`.
- The source snapshots were obtained using configured GitHub SSH credentials.
- Entry: `source/upstream/python/attn_tune.py`.
- Per-shape search: official `--backend tilelang --fusion auto --trials 64
  --population 64 --evolve-iters 1 --seed 3`, including upstream final reranking.
- Each shape has an independent tuning database. No neighboring-shape timing,
  binary, or tuning database is substituted for a missing shape.

These 25 exports will be **local upstream-tuner outputs**, not prepublished
paper exports. Sixty-four trials do not establish a global optimal configuration.
The final report will retain this provenance distinction.

The existing compatible runtime overlay contains an older libtvm.so which
shadows TVM_LIBRARY_PATH and lacks the full scheduler API. Tuning therefore
uses the unchanged upstream TVM Python source with the already compiled,
ABI-compatible Python3.13 FFI, and the complete compiler libraries under
`/work/nvme/beqg/mbudiu2/neptune-release-build2`. The required Nautilus scheduler
registration was verified before GPU tuning. Attention compute and compiler
algorithms are not locally rewritten.

## Measurement

After tuning, each new shape uses the same GPU allocation to measure all three
methods with the same saved input bytes:

1. Nautilus exported binary (`scripts/bench_paper_sweep.py`).
2. Official FlexAttention lower-right BlockMask, original max-autotune settings.
3. Official TileOPs GQADenseFwdOp, labeled **TileLang** as previously requested.

Every result must pass the existing FP32 reference and relative-L2 gate, then
10 warmups and **50 cold-L2 repeats** (200MiB flush). All GPU kernel durations
in each repeat are summed. Setup, compilation and tuning are excluded.
Nautilus/TileLang layouts and lower-right causal semantics remain the same.

## Execution and plotting

`run_missing_case.sh CASE` performs one complete tune-and-measure pipeline.
The initial successful environment probe is pilot job3479794; the remaining
array job3479801 depends on that entire pipeline succeeding before it starts.
Each missing shape will have its own logs and tuning files.

`make_plots.py` reuses the prior verification and the existing `plot_from_csv.py`.
It now rejects any Q without all nine KV lengths, any missing method, mismatched
inputs, invalid samples or incorrect kernel hash. GPU hosts must match within
each shape; reused shapes and new array shapes may come from different nodes.
It will write six complete PDF curves only after all **162** points are present.
The previous partial-coverage PDFs remain as historical artifacts.

## Final completion

The full matrix is now complete: **54/54 shapes and 162/162 method-shape pairs** passed the original validation gate. This includes all nine KV lengths for every Q. The final missing Q64/KV8192 kernel was regenerated with the local upstream tuner validation harness using `torch.empty` outputs; this caught an unwritten-output bug that the upstream zero-initialized tuner check had masked. Its replacement passed the independent reference gate and was measured with all three methods. No tolerance was relaxed.

Final PDFs are:

- [Q2](../../plots_from_csv/gh200-fp16-tc1-mtpc-Q2-full-kv-20261028.pdf)
- [Q4](../../plots_from_csv/gh200-fp16-tc1-mtpc-Q4-full-kv-20261028.pdf)
- [Q8](../../plots_from_csv/gh200-fp16-tc1-mtpc-Q8-full-kv-20261028.pdf)
- [Q16](../../plots_from_csv/gh200-fp16-tc1-mtpc-Q16-full-kv-20261028.pdf)
- [Q32](../../plots_from_csv/gh200-fp16-tc1-mtpc-Q32-full-kv-20261028.pdf)
- [Q64](../../plots_from_csv/gh200-fp16-tc1-mtpc-Q64-full-kv-20261028.pdf)

The final comparison and per-Q CSVs are in this directory. The earlier partial
`f8beba1` PDFs remain historical artifacts and are not the complete sweep.

## Slow-point audit

The four points where Nautilus is slower than the plotted TileLang/TileOPs
baseline were remeasured 100 times on one GH200 (`gh031`). The gaps persist:
Q4/L32768 5.9%, Q32/L32768 9.4%, Q64/L256 36.1%, and Q64/L32768 15.5%.
Kernel-level decomposition and resource evidence are in
[slow_points_audit.md](slow_points_audit.md).
