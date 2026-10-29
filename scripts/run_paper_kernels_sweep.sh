#!/usr/bin/env bash
# Measure every exported kernel in the current paper_kernels sweep.
set -euo pipefail
cd /u/yucheny8/argus
output="${1:?Usage: bash scripts/run_paper_kernels_sweep.sh OUTPUT_DIRECTORY}"
repeat="${REPEAT:-50}"
if [[ -e "$output" ]]; then
    echo "Refusing to reuse existing output directory: $output" >&2
    exit 2
fi
mkdir -p "$output"
output=$(realpath "$output")

# The exported modules were built with Neptune TVM/Triton, not the latest TileLang TVM.
source /u/yucheny8/miniconda3/etc/profile.d/conda.sh
set +u
conda activate /u/yucheny8/miniconda3/envs/ntawa
set -u
export PYTHONPATH=/work/nvme/beqg/yucheny8/argus-paper-kernels/runtime_py313:/u/yucheny8/argus
export TVM_LIBRARY_PATH=/work/nvme/beqg/yucheny8/src/neptune-internal-refactor/build/temp.linux-aarch64-cpython-311
export PATH=/u/yucheny8/nsight-systems-2025.5/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR=/tmp/argus-paper-sweep-triton
export CUDA_CACHE_PATH=/tmp/argus-paper-sweep-cuda
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
ulimit -c 0

if [[ -n "${CASES:-}" ]]; then
    read -r -a cases <<< "${CASES//,/ }"
else
    mapfile -t cases < <(find paper_kernels -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
        | awk '/^(prefill|prefillc|decode|sparse)_L[0-9]+$/ || /^mtp_Q[0-9]+_L[0-9]+$/ || /^mtpc_Q[0-9]+_L[0-9]+$/ || /^sparse(_split)?$/' \
        | sort -V)
fi
failed=0
for case in "${cases[@]}"; do
    echo "Running $case: $output/$case.log"
    if python -u -m scripts.bench_paper_sweep "$case" \
            --output "$output/$case" --repeat "$repeat" \
            >"$output/$case.log" 2>&1; then
        echo "Finished $case"
    else
        echo "FAILED $case (see log)" >&2
        failed=1
    fi
done
printf 'cases=%d repeat=%s failed=%d\n' "${#cases[@]}" "$repeat" "$failed" | tee "$output/summary.txt"
if (( failed )); then
    exit 1
fi
python -m scripts.summarize_paper_sweep "$output" >"$output/results.json"
