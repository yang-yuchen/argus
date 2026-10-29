#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
source /u/yucheny8/miniconda3/etc/profile.d/conda.sh
conda activate /work/nvme/beqg/yucheny8/envs/argus-gh200-latest
export PYTHONPATH=/u/yucheny8/argus:/u/yucheny8/argus/.tileops_work/src
export PATH=/u/yucheny8/nsight-systems-2025.5/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
mkdir -p profiles_official_tileops_all_20261026
for l in 256 512 1024 2048 4096 8192 16384 32768 65536; do
  python -u -m scripts.bench_official_tileops_dense mtp "$l" --output "profiles_official_tileops_all_20261026/mtp_Q16_L${l}" --repeat 50
done
for l in 256 512 1024 2048 4096 8192 16384 32768; do
  python -u -m scripts.bench_official_tileops_dense decode "$l" --output "profiles_official_tileops_all_20261026/decode_L${l}" --repeat 50
done
