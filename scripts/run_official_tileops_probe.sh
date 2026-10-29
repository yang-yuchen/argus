#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
source /u/yucheny8/miniconda3/etc/profile.d/conda.sh
conda activate /work/nvme/beqg/yucheny8/envs/argus-gh200-latest
export PYTHONPATH=/u/yucheny8/argus:/u/yucheny8/argus/.tileops_work/src
export PATH=/u/yucheny8/nsight-systems-2025.5/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
python -u -m scripts.bench_official_tileops_dense mtp 65536 --output profiles_official_tileops_probe/mtp_Q16_L65536 --repeat 50
python -u -m scripts.bench_official_tileops_dense decode 8192 --output profiles_official_tileops_probe/decode_L8192 --repeat 50
