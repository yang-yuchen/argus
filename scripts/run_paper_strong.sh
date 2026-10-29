#!/usr/bin/env bash
# One GPU allocation; Hopper-specific configurations, shared inputs and FP64/RMSE gates.
set -euo pipefail
cd /u/yucheny8/argus
output="${1:?Usage: bash scripts/run_paper_strong.sh OUTPUT_DIRECTORY [VARIANT]}"
if [[ -e "$output" ]]; then
    echo "Refusing to reuse output directory: $output" >&2
    exit 2
fi
mkdir -p "$output"
output=$(realpath "$output")
export PATH="/u/yucheny8/nsight-systems-2025.5/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
ulimit -c 0
failed=0
variants=(decode mtp sparse sparse_split)
if [[ "${2:-all}" != all ]]; then
    case "$2" in
        decode|mtp|sparse|sparse_split) variants=("$2") ;;
        *) echo "Unknown variant: $2" >&2; exit 2 ;;
    esac
fi
for variant in "${variants[@]}"; do
    echo "Nautilus and accuracy oracle: $variant"
    if ! (
        export PYTHONPATH=/work/nvme/beqg/yucheny8/argus-paper-kernels/runtime_py313:/u/yucheny8/argus
        export TVM_LIBRARY_PATH=/work/nvme/beqg/yucheny8/src/neptune-internal-refactor/build/temp.linux-aarch64-cpython-311
        libraries=/u/yucheny8/miniconda3/envs/ntawa/lib/python3.13/site-packages/nvidia
        export LD_LIBRARY_PATH="$libraries/cublas/lib:$libraries/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
        /u/yucheny8/miniconda3/envs/ntawa/bin/python -u -m scripts.bench_paper_strong \
            nautilus "$output" --variant "$variant"
    ) >"$output/${variant}_nautilus.log" 2>&1; then
        echo "FAILED Nautilus/$variant; no valid benchmark oracle" >&2
        failed=1
        continue
    fi
    if [[ "$variant" == decode || "$variant" == mtp ]]; then
        families=(pdf_dense libraries tilelang triton)
    else
        families=(sparse)
    fi
    for family in "${families[@]}"; do
        echo "Strong candidates: $variant/$family"
        if ! (
            source envs/activate_gh200_latest.sh
            export PYTHONPATH=/u/yucheny8/argus
            unset TVM_LIBRARY_PATH
            python -u -m scripts.bench_paper_strong candidates "$output" \
                --variant "$variant" --family "$family"
        ) >"$output/${variant}_${family}.log" 2>&1; then
            echo "FAILED $variant/$family; inspect candidate log" >&2
            failed=1
        fi
    done
done
exit "$failed"
