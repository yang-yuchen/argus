# Whole-step CUDA Graph verification: decode256 /32K

## Result

Whole-step capture succeeds for the aligned TileOPs integration. Once per-layer host
submission is removed from each forward, wall latency clearly increases with KV length.

Same GH200 GPU, node **gh006**, Slurm job **3473801**, dedicated latest environment.
Llama-7B backbone,32layers, FP16, batch1, query1, existing prefix length KV−1.
Both lengths and both execution modes are sampled30times in alternating/reversed order.

| KV length | Normal compiled forward | Whole-step graph replay | Normal / replay |
|---|---:|---:|---:|
|256|28.230ms|**5.103ms**|5.532×|
|32768|28.600ms|**9.558ms**|2.992×|

Replay grows by **4.455ms** from256 to32K; the normal forward grows by only0.370ms.
Replay sample standard deviations are0.033ms and0.067ms, respectively. Normal-forward
standard deviations are3.103ms and3.340ms. Host elapsed time until `graph.replay()`
returns is only **0.018ms /0.015ms**, compared with **28.197ms /28.570ms** for the
normal model call. Host-return timing includes runtime/driver work, not just pure CPU compute.

These ratios compare normal versus graph execution of the same TileOPs-backed model.
They are not Nautilus-versus-TileOPs speedups. Absolute normal timings differ from the
earlier sweep because this is a different node/run with both compiled models resident.

## What was captured

The capture surrounds the entire compiled backbone forward: new-token KV writes,
projections, attention, MLPs, normalization and residual work. Prefix construction,
input preparation, compilation, capture and correctness checks are outside formal timing.
Formal samples use device-wide synchronization before/after either full normal forward
or full graph replay; no profiler runs during these samples.

The original TileOPs attention IR/configuration is unchanged. TVM-FFI's documented
`use_torch_stream(torch.cuda.graph(...))` context aligns the prebound TVM executable
with PyTorch's capture stream. No per-layer stream setter is added to the timed path.

## Correctness and capture completeness

- Each endpoint passes independent FP32 attention and model checks before capture.
  Restored graph output relative L2 versus the independent model reference is
  **0.005604 /0.005482**, below1%.
- For each endpoint, two new random input tokens are copied into the existing input
  storage. Graph output matches normal compiled output exactly in these checks
  (reported relative L2 **0.0**).
- Historical V in the first layer is perturbed by+0.5. Model output changes substantially
  (relative change **1.246 /1.291**), and graph still matches normal output with relative
  L2 **0.0**. Input token and original prefix V are restored before timing.
- Output comparisons alone are not the capture proof, since the two modes share bound
  attention buffers. Separate replay traces verify **all GPU kernels have nonzero
  graph IDs**:642kernels per256forward and674per32Kforward.
- In particular, every replay includes **32 /64attention CUDA kernels**, all belonging
  to the graph, and **64KV index-copy kernels** (K andV for each of32layers).
- Graph traces contain32`memcpy32_post` kernels in place of the normal Q-staging copy
  operations. This accounts for32additional kernels compared with earlier normal traces;
  attention and the full model computation have not been omitted.

Profiler evidence is kept separate from formal latency samples:

| Mean profiled GPU duration sum per replay |256|32K|
|---|---:|---:|
|Attention kernels|0.185ms|4.630ms|
|All GPU kernels|5.063ms|9.544ms|

The agreement in scale with replay wall latency supports the host-overhead diagnosis.
Profiler values are diagnostic and are not substituted for the30formal wall samples.

## Artifacts and reproduction

- [`results.json`](results.json):120formal samples, correctness, GPU identity and replay profiles.
- [`validation_summary.json`](validation_summary.json):means, medians, standard deviations, ratios.
- [`L256_graph.trace.json`](L256_graph.trace.json),
  [`L32768_graph.trace.json`](L32768_graph.trace.json):raw replay traces.
- [`L256.dot`](L256.dot), [`L32768.dot`](L32768.dot):CUDA Graph debug dumps.
- Script: [`verify_decode_cudagraph_20261027.py`](../verify_decode_cudagraph_20261027.py).

Inside an allocated GHx4 GPU, activate `envs/activate_gh200_latest.sh`, set the same
HF/repo/TileOPs PYTHONPATH as the existing sweep, and run the script with a new output
directory. Set `TORCHINDUCTOR_CUDAGRAPHS=0` so this explicit whole-step capture is the
only graph optimization. The script refuses to overwrite an existing output directory.

Job completed0:0 and GPU was released. This verifies only the fixed-address, fixed-position
single-step TileOPs decode endpoints. It does not implement a dynamic generation loop,
rerun the8length sweep, or compare graph-enabled Nautilus/FlexAttention.
