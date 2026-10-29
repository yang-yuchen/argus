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
        if python - "$prefix.json" "$method" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);d=json.loads(p.read_text()) if p.exists() else {}
complete=d.get('measurement_complete') and d.get('correct')
if sys.argv[2]=='flex':complete=complete and d.get('source',{}).get('flex_search_space')=='DEFAULT' and d.get('source',{}).get('kernel_options',{}).get('BLOCK_M')==32 and d.get('source',{}).get('kernel_options',{}).get('BLOCK_N')==64 and d.get('source',{}).get('kernel_options',{}).get('num_warps')==8
sys.exit(0 if complete else 1)
PY
        then
            continue
        fi
        archive="$root/${method}_previous_attempt"
        mkdir -p "$archive"
        for extension in log json csv nsys-rep sqlite; do
            if [[ -f "$prefix.$extension" ]]; then mv "$prefix.$extension" "$archive/"; fi
        done
        python -u -m scripts.bench_sparse_official "$method" "$length" \
            --inputs "$root/nautilus" --output "$prefix" --repeat 50 \
            >"$prefix.log" 2>&1 || echo "FAILED $method L$length: $prefix.log" >&2
    done
done
if [[ "${SPARSE_FINAL_PASS:-0}" != 1 ]]; then
    SPARSE_FINAL_PASS=1 bash profiles_sparse_compilers_20261027/finish_remaining.sh
fi
