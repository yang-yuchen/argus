#!/usr/bin/env bash
# Run inside one Slurm GH200 allocation. Existing environments stay unchanged.
set -euo pipefail
cd /u/yucheny8/argus
output="${1:?Usage: bash scripts/run_paper_kernels.sh OUTPUT_DIRECTORY}"
if [[ -e "$output" ]]; then
    echo "Refusing to reuse existing output directory: $output" >&2
    exit 2
fi
mkdir -p "$output"
output=$(realpath "$output")
export PATH="/u/yucheny8/nsight-systems-2025.5/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR=/tmp/argus-paper-suite-triton
export CUDA_CACHE_PATH=/tmp/argus-paper-suite-cuda
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
ulimit -c 0
failed=0
for variant in prefill decode mtp sparse sparse_split; do
    for implementation in nautilus baseline; do
        echo "Running $variant/$implementation: $output/${variant}_${implementation}.log"
        if (
        if [[ "$implementation" == nautilus || "$variant" == prefill ]]; then
            executable=/u/yucheny8/miniconda3/envs/ntawa/bin/python
            export PYTHONPATH=/work/nvme/beqg/yucheny8/argus-paper-kernels/runtime_py313:/u/yucheny8/argus
            export TVM_LIBRARY_PATH=/work/nvme/beqg/yucheny8/src/neptune-internal-refactor/build/temp.linux-aarch64-cpython-311
        else
            source /u/yucheny8/argus/envs/activate_gh200_latest.sh
            executable=/work/nvme/beqg/yucheny8/envs/argus-gh200-latest/bin/python
            export PYTHONPATH=/u/yucheny8/argus
            unset TVM_LIBRARY_PATH
        fi
        "$executable" -u -m scripts.bench_paper_kernels "$variant" \
                --implementation "$implementation" --output "$output/${variant}_${implementation}" \
                --repeat 50
        ) >"$output/${variant}_${implementation}.log" 2>&1; then
            echo "Completed $variant/$implementation"
        else
            echo "FAILED $variant/$implementation (see log)" >&2
            failed=1
        fi
    done
done
if (( failed )); then
    exit 1
fi
PYTHONPATH=/u/yucheny8/argus /work/nvme/beqg/yucheny8/envs/argus-gh200-latest/bin/python \
    -m scripts.summarize_paper_kernels "$output" --repeat 50
