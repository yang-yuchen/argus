# Official TileLang autotuned causal MTP and MHA decode

Official source commit: `fdb19f26d61e71cabcc61abbc66ab60a97e3a100`. The vendored kernels are unchanged and match their SHA256 manifests.

## Implementations

- Decode: `examples/flash_decoding/example_gqa_decode.py`, with query heads = KV heads = 32 (`groups=32`). Actual MHA shape: B=1, Q=1, KV=L, H=32, D=128; all cached keys visible. The upstream `groups` parameter here is the KV head count.
- Causal MTP: `examples/flash_attention/example_mha_fwd_varlen.py`, B=1, Q=16, KV=L, H=4, D=128; lower-right causal condition `k <= q + (L-16)`.
- Both use upstream `@autotune` with actual Nautilus input tensors supplied through `set_autotune_inputs`. No tunable parameters are forced in the call.
- Decode searches 24 official configurations; MTP searches 32, rejecting configurations that fail to compile or benchmark.

## Measurement

Slurm job `3466413`, GH200 node `gh032`, completed successfully (exit 0). Environment: `/work/nvme/beqg/yucheny8/envs/argus-gh200-latest`.
Each selected kernel passes the independent FP32 reference (rtol=1e-2, atol=3e-3), then runs 10 warmups and 50 cold-L2 repeats. Times below sum GPU kernel durations per repeat; compilation, autotuning, input layout preparation, and CPU overhead are excluded.
The tuner uses upstream timing defaults; its internal warm-cache objective differs from the final cold-L2 measurement. Earlier fixed-configuration values are from a prior GH200 run, so speedup also includes cross-run/node variation.

## Results

| Workload | KV length | Earlier fixed (us) | Autotuned (us) | Earlier / autotuned | Selected configuration |
|---|---:|---:|---:|---:|---|
| decode | 256 | 14.584 | 8.247 | 1.768x | `{"block_H": 64, "block_N": 128, "num_split": 1, "num_stages": 3, "threads": 128}` |
| decode | 512 | 16.518 | 9.295 | 1.777x | `{"block_H": 64, "block_N": 64, "num_split": 8, "num_stages": 3, "threads": 128}` |
| decode | 1024 | 19.212 | 12.875 | 1.492x | `{"block_H": 64, "block_N": 128, "num_split": 8, "num_stages": 1, "threads": 128}` |
| decode | 2048 | 26.393 | 19.620 | 1.345x | `{"block_H": 64, "block_N": 64, "num_split": 8, "num_stages": 1, "threads": 128}` |
| decode | 4096 | 37.647 | 32.477 | 1.159x | `{"block_H": 64, "block_N": 128, "num_split": 8, "num_stages": 1, "threads": 128}` |
| decode | 8192 | 59.192 | 51.983 | 1.139x | `{"block_H": 64, "block_N": 64, "num_split": 8, "num_stages": 1, "threads": 128}` |
| decode | 16384 | 108.064 | 87.411 | 1.236x | `{"block_H": 64, "block_N": 128, "num_split": 8, "num_stages": 1, "threads": 128}` |
| decode | 32768 | 204.957 | 157.160 | 1.304x | `{"block_H": 64, "block_N": 128, "num_split": 8, "num_stages": 1, "threads": 128}` |
| causal_mtp | 256 | 7.754 | 5.654 | 1.371x | `{"block_M": 64, "block_N": 128, "num_stages": 1, "threads": 128}` |
| causal_mtp | 512 | 11.307 | 8.470 | 1.335x | `{"block_M": 64, "block_N": 128, "num_stages": 1, "threads": 128}` |
| causal_mtp | 1024 | 18.654 | 14.109 | 1.322x | `{"block_M": 64, "block_N": 128, "num_stages": 1, "threads": 128}` |
| causal_mtp | 2048 | 33.304 | 25.320 | 1.315x | `{"block_M": 64, "block_N": 128, "num_stages": 1, "threads": 128}` |
| causal_mtp | 4096 | 62.557 | 47.942 | 1.305x | `{"block_M": 64, "block_N": 128, "num_stages": 1, "threads": 128}` |
| causal_mtp | 8192 | 121.117 | 93.521 | 1.295x | `{"block_M": 64, "block_N": 128, "num_stages": 1, "threads": 128}` |
| causal_mtp | 16384 | 238.050 | 184.422 | 1.291x | `{"block_M": 64, "block_N": 128, "num_stages": 1, "threads": 128}` |
| causal_mtp | 32768 | 472.289 | 364.011 | 1.297x | `{"block_M": 64, "block_N": 128, "num_stages": 1, "threads": 128}` |
| causal_mtp | 65536 | 938.877 | 725.164 | 1.295x | `{"block_M": 64, "block_N": 128, "num_stages": 1, "threads": 128}` |

## Interpretation and limitations

For MTP, every length selects block_M=64, block_N=128, num_stages=1, threads=128. This improves the previous fixed 128x128 configuration, but Q=16 still fills only 16 rows. The kernel launches only B*H*ceil(Q/block_M)=4 CTAs and has no split-KV path; long KV is scanned serially within those CTAs.
For decode, the official GQA example is a Q=1 kernel with a tunable split-KV path. Setting KV heads equal to query heads preserves MHA semantics. Its block_H=64 is a tile over query heads within a KV group, not 64 MTP query tokens; with MHA, only one row per group is useful.
These are the best-matched official paths identified for these semantics. This study does not prove a global optimum across all TileLang kernel families or configurations outside the upstream search spaces.

## Reproduction

```bash
srun --account=beqg-dtai-gh --partition=ghx4-interactive --time=02:00:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gpus-per-node=1 \
  bash scripts/run_official_tilelang_decode.sh NEW_DECODE_OUTPUT
srun --account=beqg-dtai-gh --partition=ghx4-interactive --time=02:00:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gpus-per-node=1 \
  bash scripts/run_official_tilelang_mtpc.sh NEW_MTP_OUTPUT
```
