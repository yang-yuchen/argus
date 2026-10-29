#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
out="${1:?Usage: bash scripts/run_official_tilelang_mtpc.sh OUTPUT_DIRECTORY}"
if [[ -e "$out" ]]; then echo "Output exists: $out" >&2; exit 2; fi
mkdir -p "$out"
set +u
source envs/activate_gh200_latest.sh
set -u
export PYTHONPATH=/u/yucheny8/argus PATH=/u/yucheny8/nsight-systems-2025.5/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
for length in 256 512 1024 2048 4096 8192 16384 32768 65536; do
  python -u -m scripts.bench_official_tilelang_mtpc "$length" --output "$out/mtpc_Q16_L${length}" --repeat 50 >"$out/mtpc_Q16_L${length}.log" 2>&1
done
