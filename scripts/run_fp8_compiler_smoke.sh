#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
work=/u/yucheny8/argus/profiles_fp8_compilers_20261029
run=${1:-$work/smoke_${SLURM_JOB_ID:-manual}}
mkdir -p "$run/nautilus" "$run/tilelang"
complete() {
    [[ -f $1.json ]] || return 1
    python -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x.get("correct") and x.get("measurement_complete") and x.get("repeats")==len(x.get("samples_us",[]))==50' "$1.json"
}
for case in prefill8_L2048 prefillc8_L2048 decode8_L4096 mtp8_Q16_L4096 mtpc8_Q16_L4096; do
    export FP8_PROVIDER=nautilus
    source envs/activate_fp8_compilers.sh
    if ! complete "$run/nautilus/$case"; then
        python -u -m scripts.bench_paper_sweep "$case" --output "$run/nautilus/$case" --repeat 50 >"$run/nautilus/$case.log" 2>&1
    fi
    export FP8_PROVIDER=tilelang
    source envs/activate_fp8_compilers.sh
    if ! complete "$run/tilelang/$case"; then
        python -u -m scripts.bench_fp8_baseline "$case" --method tilelang --input-file "$run/nautilus/${case}_inputs.pt" --output "$run/tilelang/$case" --repeat 50 >"$run/tilelang/$case.log" 2>&1
    fi
    echo "FP8 SMOKE PASSED $case"
done
python -m scripts.summarize_paper_sweep "$run/nautilus"
python -m scripts.summarize_paper_sweep "$run/tilelang"
