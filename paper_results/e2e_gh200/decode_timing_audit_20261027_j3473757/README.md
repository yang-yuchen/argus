# Why decode wall latency stays flat: paired timing audit

Follow-up [whole-step CUDA Graph validation](../decode_cudagraph_20261027_j3473801/README.md)
succeeded: replay latency grows from5.103ms at256 to9.558ms at32K, with all backbone
and attention kernels verified inside the graph. This validates the proposed host-overhead
reduction for the two TileOPs endpoints.

## Result

The wall measurement includes the GPU work. In this harness, CPU/driver dispatch limits
model completion, so extra attention work largely fits into gaps in GPU task submission.
The flat curve is not evidence that attention costs are independent of KV length.

GHx4 job **3473757**, node **gh109**, same GH200 GPU for both lengths. Two independent
Llama-7B backbone models, identical weight hashes, B1/FP16, Q1, native BSHD cache and the
aligned `tileops_bound` integration. Each model has its own hook/cache. Ten alternating
warmup rounds precede 30 paired measurement rounds; AB/BA order reverses each round.

### Unprofiled alternating measurements

| Measurement | KV256 | KV32768 |
|---|---:|---:|
| Synchronized model wall mean | 25.882 ms | 25.863 ms |
| Elapsed host time until model call returns | 25.841 ms | 25.821 ms |
| CUDA event span, including idle | 25.867 ms | 25.848 ms |
| Time from model return through end event and device synchronization | 0.042 ms | 0.043 ms |

The paired long-minus-short wall difference is **-0.019 ms**, with paired sample standard
deviation **2.264 ms** and standard error **0.413 ms**. There is no resolved wall increase
in these samples. The host interval includes Python, driver calls and any implicit waits;
it is not a measurement of pure CPU computation.

Both timing boundaries use device-wide `torch.cuda.synchronize()`. This would wait for
work on any stream. The event spans agree with wall, and the traces show all GPU kernels
on stream7. The absence of a long final wait is consistent with the GPU keeping up with
the relatively slow host submission path.

### Separate profiler measurements: 2 forwards per length

| Mean per forward, with profiler | KV256 | KV32768 |
|---|---:|---:|
| Attention GPU kernel duration sum | 0.180 ms | 4.632 ms |
| All GPU kernel duration sum | 4.887 ms | 9.349 ms |
| GPU busy interval union, including memcpy/memset | 4.899 ms | 9.360 ms |
| GPU idle time between first and last GPU activity | 44.110 ms | 41.036 ms |
| GPU first-to-last activity span | 49.009 ms | 50.396 ms |
| Profiled model wall | 49.384 ms | 50.745 ms |

The actual attention computation grows by **4.452 ms**. In the profiled timeline,
GPU idle gaps shrink by about **3.074 ms**, while the activity span increases by only
about **1.387 ms**. This directly demonstrates the masking mechanism, though not every
additional GPU millisecond is hidden in these profiled samples.

Profiler instrumentation substantially increases host latency. Its absolute idle/wall
numbers must not be substituted into the unprofiled 25.9ms samples. Keeping two models
and Dynamo specializations resident also changes host guard behavior relative to the
original single-model sweep. This paired audit validates the mechanism in this harness,
not a per-component attribution for every point in the earlier curve.

## Measurement validity and benchmark limitation

- Each length passes independent FP32 attention and complete-model checks, including the
  compiled models. Actual KV shapes are256 and32768, not repeated short-KV execution.
- Profiler counts32 attention CUDA kernels for the short branch and64 for the long branch,
  matching32layers and the official short/long implementations.
- There is no missing final synchronization or unmeasured GPU tail. Prefix construction
  remains intentionally outside timing: this measures one decode step with an existing cache.
- The current per-layer disabled attention hook and model graph transitions add enough
  host overhead to obscure attention scaling. The wall numbers describe this integration;
  they should not be presented as an optimized GPU-bound decode benchmark or proof that
  attention kernel improvements have no model-level effect.
- To measure that effect, host submission/graph transitions need to be reduced and the
  resulting execution validated, for example with correctly captured whole-step CUDA graphs.
  No graph optimization was introduced in this diagnostic.

## Artifacts

- [`results.json`](results.json): all60unprofiled samples, correctness, GPU identity, and
  the two separate per-length profiler summaries.
- [`paired_statistics.json`](paired_statistics.json): paired long-minus-short statistics.
- [`L256.trace.json`](L256.trace.json), [`L32768.trace.json`](L32768.trace.json): raw traces.
- Script: `../diagnose_decode_timing_20261027.py`; runner:
  `../run_decode_timing_audit_20261027.sh`.

This audit does not remeasure other backends or the full8length sweep. Profiler records
and event span are diagnostic metrics, separate from synchronized wall samples.
