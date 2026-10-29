# MTPC slow-point optimization (complete)

Final validated comparison: [RESULTS.md](RESULTS.md) and [comparison.csv](comparison.csv). All four formal measurements are complete; the sections below retain the experiment history.

The four original slower shapes are Q4/KV32768, Q32/KV32768,
Q64/KV256 and Q64/KV32768. Original exports/results remain unchanged in
`../mtpc_full_kv_20261028/`.

## Official manual parameter sweep

Only the parameters of upstream
`neptune.operators.attn_plain.schedule_split_kv_flash` are varied: split count,
KV tile size, warp count, stages, and the existing TMA/warp-specialization controls.
Attention math and the lower-right causal mask are unchanged. No new attention
kernel is handwritten.

Each candidate is checked against the existing chunked FP32 reference on:

- The exact original input bytes.
- An independent GPU random seed17 input set.
- Outputs prefilled with NaN and with1, four cycles each, to reject incomplete
  writes and dependence on zero initialization.

The gates remain rtol1e-2, atol3e-3 and relativeL2<1e-2. Correct candidates are
screened using the upstream `_kernel_ms` cold-L2 CUPTI kernel-duration timer,
two rounds per candidate. The logs contain no event-timing fallback warnings.
This screening uses the upstream256,000,000-byte flush and500ms steady-state
warmup; final comparisons use the existing Argus200MiB/50-repeat nsys protocol.

Manual job3481225 completed all three applicable shapes on gh035. Preliminary
screening results (not final nsys comparison):

| Shape | Original us | Best manual us | Parameters |
|---|---:|---:|---|
| Q64/KV256 | 9.294 | 5.726 | splits4, tile32, warps4, stages2, TMAoff |
| Q32/KV32768 | 35.613 | 31.018 | splits32, tile64, warps4, stages2, TMAoff |
| Q64/KV32768 | 38.764 | 31.849 | splits32, tile64, warps4, stages2, TMAoff |

The best exports, original and extra-seed correctness statistics, every failed
configuration, and timings are in `manual/<case>/`.

Q4 is not directly supported by the manual function's query extent constraint.
A separate `trace_probe.py` replays the published Q4/KV65536 trace on a fresh
upstream Q4/KV32768 compute definition, adjusting only sampled KV tiling decisions.
It does not edit the mathematical compute or replace the causal offset manually.
That probe is queued as job3481333.

## Continued automatic search

Each same-shape64-trial database was copied into `retune/<case>/tilelang`.
The planned ceiling is256 committed records, with new trials run in batches of
up to64 and safely resumed from committed records during short GPU backfill.
The original kernel is always retained as an independently checked fallback.

Job array3481282 is the current automatic-search attempt. Task0 ran for10min
and committed32 additional records (96 total) before the backfill time limit.
It found a roughly29.9us Q4/KV32768 candidate. This is not yet an independently
accepted performance result. Its checkpoint is being filtered by
`select_checked.py` (job3481384). Other array tasks remain queued.

The upstream tuner times candidates but does not numerically verify every
candidate. Therefore we never accept `best/export` or the fastest raw record
without independent checks. `select_checked.py` scans records in timing order,
seeks16 independently correct candidates, and includes the original/manual
fallbacks. Failed or poisoned-output-dependent kernels are rejected. Selected
exports still require final nsys timing.

## Final candidate measurement (updated by user)

The user explicitly permits comparison across the GH200 cluster. Same-GPU
and counterbalanced repeated baseline jobs were cancelled. Job3481333 measures
the three optimized manual exports first, on any available GH200, with the
existing50-repeat cold-L2 nsys benchmark. Separate measurement array3481488 was
cancelled so these short measurements can share the first available allocation. Original and TileOPs values are reused
from the completed full sweep. The same job then probes Q4 and measures its optimized export
immediately after correctness validation. No requirement to colocate methods or
match GPU nodes is applied to this optimization comparison.

GPU resources are currently queued; scheduling estimates are not guaranteed.
The two earlier pending jobs3481218/3481224 and the unsplit verification job
3481296 were cancelled before execution in favor of checkpointable short jobs.

## Queue cleanup

Pending optional continuation tasks3481282_1/2/3 were cancelled after faster,
independently correct manual candidates were found. Their databases remain at
64 records each; Q4 has96 records. The256-record ceiling was not completed.
Only3481333 (candidate measurements and Q4 trace probe) and3481384 (independent
Q4 automatic-candidate checks) remain for this experiment. Job3481511 belongs
to a separate Agentic_Neptune experiment and was not changed.

## Actual completed jobs and remaining measurement

Jobs3481333 and3481384 completed. Q4 manual and automatic-candidate selection
chose the same independently correct export. Final50-repeat Q4 latency is
30.93888us versus saved original33.35104us and TileLang31.53792us. Input bytes
and export hashes match provenance.

Slurm used its original submitted shell snapshot, so later shell additions did
not measure the other three candidates. Job3481613 now runs only those three
measurements and then reuses scripts.summarize_paper_sweep. No baseline reruns
or extra autotuning jobs were submitted.
