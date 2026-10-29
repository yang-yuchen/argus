#!/usr/bin/env bash
# Supplement the existing decode results with only the aligned TileOPs integration.
set -euo pipefail
cd /u/yucheny8/argus
output="profiles_paper_e2e/decode_bound_only_20261027_j${SLURM_JOB_ID:?}"
if [[ -e "$output" ]]; then
    echo "Refusing to reuse output directory: $output" >&2
    exit 2
fi
test "$(git -C .tileops_work rev-parse HEAD)" = 85e7e644f1d99037e937378b8d8dc70c6568113f
mkdir -p "$output"
source envs/activate_gh200_latest.sh
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/u/yucheny8/argus/.paper_e2e_hf:/u/yucheny8/argus:/u/yucheny8/argus/.tileops_work/src
unset TVM_LIBRARY_PATH
nvidia-smi --query-gpu=name,uuid --format=csv
python -u -m scripts.bench_decode_e2e_sweep tileops_bound "$output" --repeat 30 --warmup 5 \
    >"$output/tileops_bound.log" 2>&1
