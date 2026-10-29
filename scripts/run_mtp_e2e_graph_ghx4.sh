#!/usr/bin/env bash
# Compare all three MTP attention backends using whole-step replay on one GH200.
set -euo pipefail
cd /u/yucheny8/argus
output="${1:?Usage: bash scripts/run_mtp_e2e_graph_ghx4.sh OUTPUT_DIRECTORY}"
if [[ -e "$output" ]]; then
    echo "Refusing to reuse output directory: $output" >&2
    exit 2
fi
test "$(git -C .tileops_work rev-parse HEAD)" = 85e7e644f1d99037e937378b8d8dc70c6568113f
mkdir -p "$output"
source envs/activate_gh200_latest.sh
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0
export PYTORCH_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_CUDAGRAPHS=0
baseline_path=/u/yucheny8/argus/.paper_e2e_hf:/u/yucheny8/argus:/u/yucheny8/argus/.tileops_work/src
export PYTHONPATH="$baseline_path"
unset TVM_LIBRARY_PATH
nvidia-smi --query-gpu=name,uuid --format=csv
for implementation in tileops flex nautilus; do
    if [[ "$implementation" == nautilus ]]; then
        export PYTHONPATH=/work/nvme/beqg/yucheny8/argus-paper-kernels/runtime_py313:$baseline_path
        export TVM_LIBRARY_PATH=/work/nvme/beqg/yucheny8/src/neptune-internal-refactor/build/temp.linux-aarch64-cpython-311
    fi
    python -u -m scripts.bench_paper_e2e mtpc --implementation "$implementation" \
        --kv-length 65536 --cudagraph --warmup 5 --repeat 30 \
        --output "$output/mtpc_${implementation}" >"$output/mtpc_${implementation}.log" 2>&1
done
python -m scripts.summarize_mtp_e2e_graph "$output"
