#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
output="${1:?Usage: bash scripts/run_official_helion_prefill.sh OUTPUT_DIRECTORY}"
if [[ -e "$output" ]]; then echo "Output exists: $output" >&2; exit 2; fi
mkdir -p "$output"
source /u/yucheny8/miniconda3/etc/profile.d/conda.sh
set +u
conda activate /work/nvme/beqg/yucheny8/envs/argus-gh200-latest
set -u
export PYTHONPATH=/u/yucheny8/argus
export PATH=/u/yucheny8/nsight-systems-2025.5/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
ulimit -c 0
for length in 256 512 1024 2048 4096 8192 16384 32768; do
  python -u -m scripts.bench_official_helion_prefill "$length" \
    --output "$output/prefillc_L${length}" --repeat 50 \
    >"$output/prefillc_L${length}.log" 2>&1
done
