#!/usr/bin/env bash
# Latest environment and official reference kernels; prefill is excluded.
set -euo pipefail
cd /u/yucheny8/argus
output="${1:?Usage: bash scripts/run_paper_kernels_latest.sh OUTPUT_DIRECTORY}"
if [[ -e "$output" ]]; then
    echo "Refusing to reuse existing output directory: $output" >&2
    exit 2
fi
mkdir -p "$output"
output=$(realpath "$output")
source envs/activate_gh200_latest.sh
export PYTHONPATH=/u/yucheny8/argus
unset TVM_LIBRARY_PATH
export PATH="/u/yucheny8/nsight-systems-2025.5/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
variants=(decode mtp sparse sparse_split)
failed=0
for variant in "${variants[@]}"; do
    # Reuse the previous run's input bytes so version comparisons use identical data.
    cp "profiles_paper_kernels/run_20261025_02/${variant}_inputs.pt" "$output/"
    for implementation in nautilus baseline; do
        echo "Running $variant/$implementation: $output/${variant}_${implementation}.log"
        if python -u -m scripts.bench_paper_kernels "$variant" \
                --implementation "$implementation" --output "$output/${variant}_${implementation}" \
                --repeat 50 --skip-unavailable-nautilus \
                >"$output/${variant}_${implementation}.log" 2>&1; then
            echo "Finished $variant/$implementation"
        else
            echo "FAILED $variant/$implementation (see log)" >&2
            failed=1
        fi
    done
done
if (( failed )); then
    exit 1
fi
python -m scripts.summarize_paper_kernels "$output" --repeat 50 \
    --variants "${variants[@]}" --allow-skipped-nautilus
