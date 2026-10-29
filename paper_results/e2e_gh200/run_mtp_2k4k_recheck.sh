#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
output=profiles_paper_e2e/mtpc_2k4k_recheck_j3481758
mkdir -p "$output"
source envs/activate_gh200_latest.sh
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0
export PYTORCH_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_CUDAGRAPHS=0
base=/u/yucheny8/argus/.paper_e2e_hf:/u/yucheny8/argus:/u/yucheny8/argus/.tileops_work/src
export PYTHONPATH="$base"
unset TVM_LIBRARY_PATH
nvidia-smi --query-gpu=name,uuid --format=csv >"$output/gpu.csv"
for implementation in nautilus flex tileops; do
    export PYTHONPATH="$base"
    unset TVM_LIBRARY_PATH
    if [[ "$implementation" == nautilus ]]; then
        export PYTHONPATH=/work/nvme/beqg/yucheny8/argus-paper-kernels/runtime_py313:$base
        export TVM_LIBRARY_PATH=/work/nvme/beqg/yucheny8/src/neptune-internal-refactor/build/temp.linux-aarch64-cpython-311
    fi
    python -u profiles_paper_e2e/rerun_mtp_2k4k.py "$implementation" "$output" \
        --warmup 5 --repeat 30 >"$output/${implementation}.log" 2>&1
done
