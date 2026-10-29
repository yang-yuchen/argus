#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
output="${1:?Usage: bash scripts/run_gh200_compare.sh OUTPUT_DIRECTORY}"
repeat="${REPEAT:-50}"
if [[ -e "$output" ]]; then
    echo "Refusing to reuse existing output directory: $output" >&2
    exit 2
fi
mkdir -p "$output"
output=$(realpath "$output")

source /u/yucheny8/miniconda3/etc/profile.d/conda.sh
set +u
conda activate /work/nvme/beqg/yucheny8/envs/argus-gh200-latest
set -u
export PYTHONPATH=/u/yucheny8/argus
export PATH=/u/yucheny8/nsight-systems-2025.5/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
ulimit -c 0

for kind in prefill_causal decode_causal; do
    for length in 256 512 1024 2048 4096 8192 16384 32768; do
        shape="1,$length"
        stem="$output/${kind}_L${length}"
        echo "Running $kind L$length: $stem.log"
        if python -u -m scripts.bench_gh200_compare "$kind" "$shape" \
                --profiler nsys --profile-dir "$output" --repeat "$repeat" \
                >"$stem.log" 2>&1; then
            echo "Finished $kind L$length"
        else
            echo "FAILED $kind L$length (see $stem.log)" >&2
            exit 1
        fi
    done
done
