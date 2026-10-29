#!/usr/bin/env bash
# Run the complete decode row on one GH200, using the dedicated latest baseline stack.
set -euo pipefail
cd /u/yucheny8/argus
output="${1:?Usage: bash scripts/run_decode_e2e_sweep_ghx4.sh OUTPUT_DIRECTORY}"
if [[ -e "$output" ]]; then
    echo "Refusing to reuse output directory: $output" >&2
    exit 2
fi
test "$(git -C .tileops_work rev-parse HEAD)" = 85e7e644f1d99037e937378b8d8dc70c6568113f
mkdir -p "$output"
output=$(realpath "$output")
source envs/activate_gh200_latest.sh
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
baseline_path=/u/yucheny8/argus/.paper_e2e_hf:/u/yucheny8/argus:/u/yucheny8/argus/.tileops_work/src
export PYTHONPATH="$baseline_path"
unset TVM_LIBRARY_PATH
for implementation in tileops flex nautilus; do
    if [[ "$implementation" == nautilus ]]; then
        export PYTHONPATH=/work/nvme/beqg/yucheny8/argus-paper-kernels/runtime_py313:$baseline_path
        export TVM_LIBRARY_PATH=/work/nvme/beqg/yucheny8/src/neptune-internal-refactor/build/temp.linux-aarch64-cpython-311
    fi
    echo "Running eight decode lengths: $implementation"
    python -u -m scripts.bench_decode_e2e_sweep "$implementation" "$output" \
        >"$output/${implementation}.log" 2>&1
done
python -m scripts.summarize_decode_e2e "$output"
