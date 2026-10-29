#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
root=profiles_paper_kernels/mtpc_f8beba1_20261028
export ARGUS_PAPER_KERNEL_ROOT=/u/yucheny8/argus/$root/kernels
cases=$(paste -sd, "$root/cases.txt")
if [[ ! -d "$root/nautilus" ]]; then
    REPEAT=50 CASES="$cases" bash scripts/run_paper_kernels_sweep.sh "$root/nautilus"
else
    source /u/yucheny8/miniconda3/etc/profile.d/conda.sh
    set +u
    conda activate /u/yucheny8/miniconda3/envs/ntawa
    set -u
    export PYTHONPATH=/work/nvme/beqg/yucheny8/argus-paper-kernels/runtime_py313:/u/yucheny8/argus
    export TVM_LIBRARY_PATH=/work/nvme/beqg/yucheny8/src/neptune-internal-refactor/build/temp.linux-aarch64-cpython-311
    export PATH=/u/yucheny8/nsight-systems-2025.5/bin:$PATH
    export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
    ulimit -c 0
    mkdir -p "$root/nautilus"
    while read -r case; do
        prefix="$root/nautilus/$case"
        if [[ -f "$prefix.json" ]] && python - "$prefix.json" <<'PY2'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
sys.exit(0 if d.get('correct') and d.get('measurement_complete') else 1)
PY2
        then continue; fi
        for ext in json csv nsys-rep sqlite log; do
            if [[ -f "$prefix.$ext" ]]; then
                mkdir -p "$root/nautilus_attempts/$case"
                mv "$prefix.$ext" "$root/nautilus_attempts/$case/"
            fi
        done
        python -u -m scripts.bench_paper_sweep "$case" --output "$prefix" --repeat 50 \
            >"$prefix.log" 2>&1 || echo "FAILED nautilus $case" >&2
    done < "$root/cases.txt"
fi
set +u
source envs/activate_gh200_latest.sh
set -u
export PYTHONPATH=/u/yucheny8/argus:/u/yucheny8/argus/.tileops_work/src
export PATH=/u/yucheny8/nsight-systems-2025.5/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
unset TVM_LIBRARY_PATH
ulimit -c 0
for method in flex tileops; do
    mkdir -p "$root/$method"
    while read -r case; do
        q=${case#mtpc_Q}; q=${q%%_L*}
        length=${case##*_L}
        prefix="$root/$method/$case"
        if [[ -f "$prefix.json" ]] && python - "$prefix.json" <<'PY2'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
sys.exit(0 if d.get('correct') and d.get('measurement_complete') else 1)
PY2
        then continue; fi
        for ext in json csv nsys-rep sqlite log; do
            if [[ -f "$prefix.$ext" ]]; then
                mkdir -p "$root/${method}_attempts/$case"
                mv "$prefix.$ext" "$root/${method}_attempts/$case/"
            fi
        done
        echo "Running $method $case"
        if [[ "$method" == flex ]]; then
            python -u -m scripts.bench_official_flex_mtpc "$length" \
                --q-length "$q" --input-file "$root/nautilus/${case}_inputs.pt" \
                --output "$prefix" --repeat 50 >"$prefix.log" 2>&1 \
                || echo "FAILED $method $case" >&2
        else
            python -u -m scripts.bench_official_tileops_dense mtp "$length" \
                --q-length "$q" --input-file "$root/nautilus/${case}_inputs.pt" \
                --output "$prefix" --repeat 50 >"$prefix.log" 2>&1 \
                || echo "FAILED $method $case" >&2
        fi
    done < "$root/cases.txt"
done
python "$root/summarize.py"
