#!/usr/bin/env bash
# Full-length FP8 decode8 / mtpc8 sweep (Nautilus vs official TileOPs "TileLang"),
# using kernels from neptune-internal nautilus-gh200 4e06b84.
set -euo pipefail
cd /u/yucheny8/argus
work=/u/yucheny8/argus/profiles_fp8_compilers_20261029
run=${1:-$work/sweep_4e06b84_${SLURM_JOB_ID:-manual}}
kernels=$work/source_4e06b84/paper_kernels
mkdir -p "$run/nautilus" "$run/tilelang"
complete() {
    [[ -f $1.json ]] || return 1
    python -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x.get("correct") and x.get("measurement_complete") and x.get("repeats")==len(x.get("samples_us",[]))==50' "$1.json"
}
nvidia-smi --query-gpu=name,uuid --format=csv > "$run/gpu.csv"
cases=()
for L in 256 512 1024 2048 4096 8192 16384 32768; do cases+=("decode8_L$L"); done
for L in 256 512 1024 2048 4096 8192 16384 32768 65536; do cases+=("mtpc8_Q16_L$L"); done
for case in "${cases[@]}"; do
    export FP8_PROVIDER=nautilus
    source envs/activate_fp8_compilers.sh
    export ARGUS_PAPER_KERNEL_ROOT=$kernels
    if ! complete "$run/nautilus/$case"; then
        python -u -m scripts.bench_paper_sweep "$case" --output "$run/nautilus/$case" --repeat 50 >"$run/nautilus/$case.log" 2>&1 || { echo "NAUTILUS FAILED $case"; continue; }
    fi
    export FP8_PROVIDER=tilelang
    source envs/activate_fp8_compilers.sh
    export ARGUS_PAPER_KERNEL_ROOT=$kernels
    if ! complete "$run/tilelang/$case"; then
        python -u -m scripts.bench_fp8_baseline "$case" --method tilelang --input-file "$run/nautilus/${case}_inputs.pt" --output "$run/tilelang/$case" --repeat 50 >"$run/tilelang/$case.log" 2>&1 || { echo "TILELANG FAILED $case"; continue; }
    fi
    echo "FP8 SWEEP PASSED $case"
done
python -m scripts.summarize_paper_sweep "$run/nautilus" || true
python -m scripts.summarize_paper_sweep "$run/tilelang" || true
