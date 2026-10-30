#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
output="${1:?Usage: bash scripts/run_mtpc_graph_sweep_ghx4.sh OUTPUT_DIRECTORY [QUERY_LENGTH]}"
query_length="${2:-16}"
case "$query_length" in 2|4|8|16) ;; *) echo "Unsupported Q=$query_length" >&2; exit 2 ;; esac
if [[ -e "$output" ]]; then
    echo "Refusing to reuse output directory: $output" >&2
    exit 2
fi
test "$(git -C .tileops_work rev-parse HEAD)" = 85e7e644f1d99037e937378b8d8dc70c6568113f
mkdir -p "$output"
# Use the same environment directly; conda activation scans stalled shared metadata.
export CONDA_PREFIX=/work/nvme/beqg/yucheny8/envs/argus-gh200-latest
export PATH="$CONDA_PREFIX/bin:$PATH"
source <(sed '/^source .*conda.sh$/d; /^conda activate /d' envs/activate_gh200_latest.sh)
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0
export PYTORCH_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_CUDAGRAPHS=0
base=/u/yucheny8/argus/.paper_e2e_hf:/u/yucheny8/argus:/u/yucheny8/argus/.tileops_work/src
export PYTHONPATH="$base"
unset TVM_LIBRARY_PATH
nvidia-smi --query-gpu=name,uuid --format=csv
for implementation in tileops flex nautilus; do
    if [[ "$implementation" == nautilus ]]; then
        export PYTHONPATH=/work/nvme/beqg/yucheny8/argus-paper-kernels/runtime_py313:$base
        export TVM_LIBRARY_PATH=/work/nvme/beqg/yucheny8/src/neptune-internal-refactor/build/temp.linux-aarch64-cpython-311
    fi
    python -u -m scripts.bench_mtpc_e2e_graph_sweep "$implementation" "$output" --query-length "$query_length" \
        >"$output/${implementation}.log" 2>&1
done
