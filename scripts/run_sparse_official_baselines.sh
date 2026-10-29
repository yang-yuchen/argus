#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
set +u
source envs/activate_gh200_latest.sh
set -u
export PYTHONPATH=/u/yucheny8/argus:/u/yucheny8/argus/.tileops_work/src
unset TVM_LIBRARY_PATH
export PATH=/u/yucheny8/nsight-systems-2025.5/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
root=profiles_sparse_compilers_20261027
for method in tilelang flex helion; do
    mkdir -p "$root/$method"
    for length in 256 512 1024 2048 4096 8192 16384 32768; do
        prefix="$root/$method/sparse_L${length}"
        if ! python -u -m scripts.bench_sparse_official "$method" "$length" \
            --inputs "$root/nautilus" --output "$prefix" --repeat 50 \
            >"$prefix.log" 2>&1; then
            echo "FAILED $method L$length: $prefix.log" >&2
        fi
    done
done
# Resume only unfinished cases, preserving prior attempts and final tuning settings.
bash scripts/run_sparse_compiler_resume.sh
