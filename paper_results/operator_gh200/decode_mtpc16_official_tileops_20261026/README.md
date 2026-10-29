# Official TileOPs dense attention results

Pinned source: `tile-ai/TileOPs` commit `85e7e644f1d99037e937378b8d8dc70c6568113f`. Public API: `GQADenseFwdOp(is_causal=True, target=BUILTIN)` with BSHD tensors; equal Q/KV head counts give MHA.

All 17 cases passed the independent FP32 reference and used 10 warmups plus 50 cold-L2 repeats. Compilation and dispatch are outside the measured region.

## Main observations

- Causal MTP dispatches to `GQADensePersistentWSKernel`, the SM90 persistent warp-specialized split-KV kernel.
- At KV=65536, TileOPs is 53.316 us, versus Nautilus 49.551 us, FlexAttention 57.153 us, and the previous official TileLang varlen kernel 725.164 us.
- At KV=256, TileOPs is 6.788 us; persistent-kernel setup/merge cost makes the previous varlen kernel faster (5.654 us). From KV=512 onward TileOPs is faster than that varlen implementation.
- Decode dispatches to `GQADecodeBs1Kernel` at every tested length. At KV=8192 it is 49.773 us, versus 50.119 us for the previous local TileLang runner and 51.983 us for the official TileLang GQA example. The improvement over the old local runner is small because that runner already used a similar Q=1 split-KV design.

See `comparison.csv` for all values and selected dispatch kernels.

## Compatibility

TileOPs declares `torch<2.14` and `tilelang==0.1.12`; this run used the existing Torch 2.14.1/cu130 and TileLang 0.1.15 environment with TileOPs installed using `--no-deps`. All kernels compiled and passed, but this is a source/API compatibility check rather than a supported dependency installation.
