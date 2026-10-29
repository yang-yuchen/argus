# Nautilus slower points audit

## Confirmed points

A same-GPU 100-repeat sequential probe (job `3481000`) was run on `gh031` for the four points where the
50-repeat sweep had Nautilus slower than the plotted TileLang label. Inputs and
Nautilus kernel hashes match the main sweep. All outputs passed correctness.

| Case | Nautilus mean (us) | TileOPs mean (us) | Nautilus slower |
|---|---:|---:|---:|
| Q4 / L32768 | 33.3027 | 31.4480 | 5.90% |
| Q32 / L32768 | 35.3574 | 32.3315 | 9.36% |
| Q64 / L256 | 9.2576 | 6.8035 | 36.07% |
| Q64 / L32768 | 38.1558 | 33.0451 | 15.47% |

The Q64/L256 TileOPs samples have a high CV (10.8%) because of one slow
sample, but Nautilus median latency is still 36.97% higher than TileOPs. The other three gaps
remain with low variation.

## Kernel structure evidence

The nsys CSVs show the following per-repeat kernel decomposition:

| Case | Nautilus kernels | TileOPs kernels |
|---|---|---|
| Q4/L32768 | `main_kernel_kernel` 31.60us + `main_kernel_1_kernel` 1.75us | `parts_kernel` 29.31us + `main_kernel` 2.22us |
| Q32/L32768 | 31.93us + 3.33us | 29.57us + 2.78us |
| Q64/L256 | 7.70us + 1.53us | one `main_kernel` 6.84us |
| Q64/L32768 | 33.06us + 5.11us | `parts_kernel` 30.14us + `main_kernel` 3.20us |

For the long KV cases, TileOPs' persistent `parts_kernel` is 2.0–2.9us
faster than Nautilus' main kernel, while its finishing kernel is also faster
for Q32/Q64. The profiler records 240 registers/thread for Nautilus Q4/Q64
long-KV main kernels, and 127 for Q32, versus 168 for TileOPs `parts_kernel`; TileOPs uses about
225KiB dynamic shared memory. This is evidence of a different occupancy/resource
tradeoff, not an isolated proof that register pressure alone causes the gap.
No hardware-counter occupancy/bandwidth ablation was performed.

Q64/L256 is different: Nautilus launches an attention kernel plus a reduction
kernel while TileOPs completes the work in one kernel. The fixed second-kernel
cost is large relative to this short workload. At Q64/L512 and longer lengths
Nautilus is faster again, so this is a short-sequence crossover rather than a
universal Nautilus regression.

## Tuning interpretation

The Nautilus winners were selected by the official upstream tuner with 42–53
successful candidates out of 64 trials on these points. This is not enough to
claim a global optimum. The Q64/L32768 manual schedule is actually 37.26us
versus the tuned 38.68us (about 3.7% faster), while Q32 manual is slower;
these manual numbers come from upstream tuning logs and have not been separately
accepted by our independent empty-output correctness gate, so they were not
substituted into the figure;
Q4 manual does not apply because its query extent is not aligned to the manual
schedule constraint. Q64/L256 manual is invalid because the split/tile
constraints do not divide KV. Thus the data support “TileOPs' official dispatch wins these shapes,” not a
claim that Nautilus is globally tuned optimally or that the compiler is always
slower.

The TileOPs curve is the same official `GQADenseFwdOp` baseline labeled
TileLang per the experiment's established naming. No kernel source was changed.

The schedule explains the short-sequence crossover more precisely. Q64/L256
uses `blockIdx.z=4`, `blockIdx.y=64` (one query row per block), and
`blockIdx.x=8`, producing 2048 first-stage blocks. It uses two warps, no TMA,
and scalar `T.tile_reduce` QK work. Q64/L512 instead uses four head blocks,
eight split blocks, a 64-row query tile, `T.tile_dot` and TMA/mbarrier; its
first stage is about 4.52us versus Q64/L256's 7.73us. Q32/L256 shows the same
pattern with 1024 first-stage blocks and scalar reduction.

This is a shape-specific tuner choice. A useful follow-up ablation is to
replay the official search while requiring the 64-row query tile or comparing
the split-KV/TMA candidate; it is not safe to change the winner in the reported
curve without rerunning correctness and the full timing protocol.

For the long-KV points, the selected block split changes with Q and length.
At L32768 the Nautilus first-stage split is Q4=64, Q16=32, Q32=128 and Q64=128;
at L65536 it changes to Q4=64, Q16=1, Q32=32 and Q64=32. Q32/Q64 also
use fewer registers at L65536 and become faster relative to TileOPs. Q16/L32768
is a control point where Nautilus wins (29.51us vs TileOPs 32.28us). This
shape-dependent reversal supports a tuning/resource configuration explanation,
not an inherent algorithmic disadvantage. Exact occupancy or memory bottleneck
was not isolated by an ablation.

The probe ran the four Nautilus cases first and the four TileOPs cases second.
It confirms that the observed gaps persist with 100 samples, identical inputs,
unchanged kernels and a common GPU; it is not an interleaved or counterbalanced
order experiment. The main sweep/PDF numeric values were not overwritten.
