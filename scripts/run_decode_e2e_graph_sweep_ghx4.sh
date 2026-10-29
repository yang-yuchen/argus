#!/usr/bin/env bash
# Run one decode backend's curve with whole-step graph replay.
set -euo pipefail
cd /u/yucheny8/argus
output="${1:?Usage: bash scripts/run_decode_e2e_graph_sweep_ghx4.sh OUTPUT_DIRECTORY}"
implementation="${2:-tileops_bound}"
case "$implementation" in
    tileops_bound|nautilus|flex) ;;
    *) echo "Expected tileops_bound, nautilus or flex as second argument" >&2; exit 2 ;;
esac
if [[ -e "$output" ]]; then
    echo "Refusing to reuse output directory: $output" >&2
    exit 2
fi
if [[ "$implementation" == tileops_bound ]]; then
    test "$(git -C .tileops_work rev-parse HEAD)" = 85e7e644f1d99037e937378b8d8dc70c6568113f
fi
mkdir -p "$output"
source envs/activate_gh200_latest.sh
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0
export PYTORCH_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_CUDAGRAPHS=0
export PYTHONPATH=/u/yucheny8/argus/.paper_e2e_hf:/u/yucheny8/argus:/u/yucheny8/argus/.tileops_work/src
unset TVM_LIBRARY_PATH
if [[ "$implementation" == nautilus ]]; then
    export PYTHONPATH=/work/nvme/beqg/yucheny8/argus-paper-kernels/runtime_py313:$PYTHONPATH
    export TVM_LIBRARY_PATH=/work/nvme/beqg/yucheny8/src/neptune-internal-refactor/build/temp.linux-aarch64-cpython-311
fi
nvidia-smi --query-gpu=name,uuid --format=csv
python -u -m scripts.bench_decode_e2e_sweep "$implementation" "$output" \
    --cudagraph --repeat 30 --warmup 5 >"$output/${implementation}.log" 2>&1
python -m scripts.summarize_decode_e2e_graph "$output" --implementation "$implementation"
