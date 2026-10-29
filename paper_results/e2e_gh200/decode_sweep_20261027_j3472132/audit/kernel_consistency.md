# Decode baseline consistency audit

Follow-up: the [controlled 32K prebound integration experiment](../../decode_bound_20261027_j3473082/README.md)
reduced TileOPs model latency from 27.885ms to 25.262ms, matching Nautilus's 25.279ms
on the same GPU. The findings below describe the original public-API integration.

## Baseline identity

The kernel plot's `TileLang` legend is the official **TileOPs** baseline, as documented in
`profiles_paper_kernels/plot_update_b7749f3_20261027/README.md`. All eight
`TileLangOfficial` CSV values exactly match the corresponding
`profiles_official_tileops_all_20261026/decode_L*.json` measurements.

For all eight lengths, kernel and model records match on repository, commit, public API,
layout, selected kernel, configuration, and input shapes. The long-sequence implementation is
`GQADecodeBs1Kernel`, commit `85e7e644f1d99037e937378b8d8dc70c6568113f`, configuration
`block_M=64, block_N=128, threads=160`. The model uses resident BSHD KV storage and records
zero full-KV staging. No replacement with a slower, different TileLang kernel was found.

## Where the model gap appears

Existing same-GPU 32K traces were reanalyzed; no new GPU measurements were performed.
GPU events below are restricted to the two annotated model forward ranges, excluding the
three subsequent correctness kernels. Each forward has 32 attention calls, two CUDA
kernels per attention call, and 642 GPU kernels in total for either implementation.

| Measurement per forward | Nautilus | TileOPs |
|---|---:|---:|
| Attention GPU kernel duration sum, profiled | 4.616 ms | 4.717 ms |
| All GPU kernel duration sum, profiled | 9.340 ms | 9.401 ms |
| Synchronized model wall latency, original unprofiled 20 samples | 23.658 ms | 31.342 ms |

Attention GPU time differs by only 0.100 ms across all 32 layers, while original model wall
time differs by 7.684 ms. This agrees with the near-parity in the kernel plot. The existing
model speedup must not be described as a comparable speedup of attention CUDA execution.

The integration paths have different host costs:

- Nautilus binds DLPack arguments and output once, then invokes its exported entry directly
  (`argus/paper_baselines/workloads.py`, `build_exported`).
- TileOPs calls the unchanged public `GQADenseFwdOp` API on every attention invocation,
  including its custom-op dispatch, cached plan handling, and output/workspace allocation.
  Across two traced forwards it has 194 `aten::empty` events versus Nautilus's 2, consistent
  with three extra allocations for each of 64 attention calls.
- Both use the same disabled attention hook. Both traces contain 580 Dynamo cache lookups;
  these traces do not establish extra graph breaks for TileOPs. Both have 64 `aten::copy_`
  events and no `aten::contiguous` events, consistent with no repeated full-KV conversion.

The TileOPs custom CPU op spans 14.980 ms across the two profiled forwards. This is an
inclusive, profiler-instrumented duration, not an isolated net overhead measurement.
Profiling changes wall latency substantially. It cannot assign the original 7.684 ms
gap precisely among dispatch, allocation, Python work, and scheduling gaps.

## Interpretation

The current curves compare synchronized model latency using the **public TileOPs API**
against a **prebound Nautilus export**. They are valid for those integration paths but do
not isolate the effect of kernel arithmetic. A controlled prebound TileOPs integration or
matched CUDA graph experiment would be needed before attributing model-level gains to
the kernels themselves. No such additional experiment is included in this audit.

Machine-readable provenance checks and forward-scoped trace totals:
[`kernel_consistency.json`](kernel_consistency.json).
