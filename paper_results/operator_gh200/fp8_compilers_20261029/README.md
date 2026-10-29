# FP8 compiler comparison setup

See [environment and reproduction instructions](../envs/paper_fp8.md).

- Nautilus source:276020e7;12 exported kernels, exact original binaries retained.
- Official TileOPs source:2c54caec; label TileLang; nativeFP8, directFP16 output. The official dedicated Dense FP8 decode kernel fails reduction lowering for the measured MHA shape; decode uses the verified public Varlen API instead.
- FlexAttention and other compiler audits: vendor manifests and other_official_audit.md.
- CPU regression checks:50 passed.
- GPU smoke job3482373 captured Nautilus prefill successfully; collector error fixed and original capture recovered. The old new-baseline environment failed official lowering (TileLang0.1.15); pinned-environment continuation3482523 now validates the remaining methods/cases in the same output directory.
- Full sequence-length measurements have not been launched by this setup task.

Official version pins are installed in workspace .fp8-baseline-env. The original
argus-gh200-latest environment was not modified. See environment_verified.json
and environment_freeze.txt. GPU validation status:

Five representative workload pairs are complete and passed the unchanged FP8
reference gate, NaN/one output checks, and complete50-repeat measurements:

| Case | Nautilus (us) | TileLang (us) | TileLang/Nautilus |
|---|---:|---:|---:|
| prefill8 L2048 | 118.84 | 102.48 | 0.86x |
| prefillc8 L2048 | 101.62 | 109.09 | 1.07x |
| decode8 L4096 | 20.40 | 50.04 | 2.45x |
| mtp8 Q16/L4096 | 6.87 | 49.05 | 7.14x |
| mtpc8 Q16/L4096 | 6.86 | 42.57 | 6.20x |

The decode TileLang result uses the official public Varlen API and selected
`GQAVarlenFP8WSFwdKernel`; the dedicated Dense FP8 decode route failed its
unchanged official ReduceOp layout lowering and is retained under failures/.
The other TileLang rows selected `GQADenseFP8Kernel`. These are smoke pairs from
separate GH200 allocations, so the ratios are directional infrastructure
checks, not a counterbalanced final comparison. Full12-case sequence sweep has
not been launched.

