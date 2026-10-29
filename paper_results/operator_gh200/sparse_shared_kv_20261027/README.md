# FlexAttention / Helion shared-KV replay

This run repeats the eight sparse sequence lengths with V as a storage-sharing
view of KV's first 512 columns (sequence stride 576, last-dimension stride 1).
It uses the existing `scripts/bench_sparse_official.py` through
`scripts/run_sparse_compiler_resume.sh flex helion` in `argus-gh200-latest`.

Only FlexAttention and Helion are rerun. The `nautilus` and `tilelang` directories
link to the existing experiment; their measurements and kernels are unchanged.
Saved inputs, FP64 validation, 10 warmups, 50 cold-L2 measurements, and the old
tuning settings are retained: Flex M32/N64/8warps with official stage search;
Helion full effort with a 180-second soft budget per length. This replay fixes
the input storage layout; it does not broaden either tuning search.

After all 16 rerun points passed the existing result checks and the shared-storage
check, the existing summary and `plot_from_csv.py` produced the
[updated PDF](../plots_from_csv/gh200-bf16-tc1-sparse-d35a206-shared-kv.pdf).

Previous copied-V data remain in `../profiles_sparse_compilers_20261027/`.
Performance changes versus that run may include node and tuning variation;
they are not an isolated paired measurement of the storage-layout change.

## Completed results

Slurm job `3473409` completed with exit code `0:0` in 36m41s on `gh093`, using
one GH200 GPU, 8 CPU cores and 64 GiB requested memory. Backfill granted a
49-minute allocation; no continuation was needed. Both methods' eight points
passed independent FP64 correctness, saved-input hashes, shared-storage metadata,
50 finite positive samples, and consistent means. Submission source hashes also
match the files used for this replay. CPU regression checks: 9 passed.

| Sequence length | FlexAttention ms | Helion ms |
|---:|---:|---:|
| 256 | 0.034851 | 0.039739 |
| 512 | 0.109820 | 0.103549 |
| 1024 | 0.316201 | 0.343366 |
| 2048 | 0.994143 | 1.128782 |
| 4096 | 3.982507 | 4.923362 |
| 8192 | 15.369576 | 19.514386 |
| 16384 | 60.578492 | 76.748531 |
| 32768 | 237.011928 | 313.237165 |

Per-point CV ranges from 0.13% to 2.62%. The full old/new comparison, including
host, CV and relative L2 error, is in
[layout_replay_comparison.csv](layout_replay_comparison.csv). The plot's source
latencies are in [plot_input.csv](plot_input.csv); higher plotted relative
performance means faster (`Nautilus latency / baseline latency`). Nautilus and
TileLang values are reused unchanged.

At 32K, the old/new means are Flex 243.759/237.012 ms and Helion
353.181/313.237 ms, latency changes of -2.77% and -11.31%. These are observations
across runs, not isolated causal estimates for the storage change. Helion's
stochastic search selected different configurations at some lengths: at L4096,
the old result used blocks32x128/stages8/warps8/persistent_interleaved, while the
new result selected blocks32x32/stages1/warps4/flat and was 20.03% slower.
Neither the fixed Flex geometry nor Helion's 180-second budget was expanded.
