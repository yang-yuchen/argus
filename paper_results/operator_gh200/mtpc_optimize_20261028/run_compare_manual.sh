#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
opt=/u/yucheny8/argus/profiles_paper_kernels/mtpc_optimize_20261028
root=/u/yucheny8/argus/profiles_paper_kernels/mtpc_full_kv_20261028
cases=${COMPARE_CASES:-"mtpc_Q64_L256 mtpc_Q32_L32768 mtpc_Q64_L32768"}
for case in $cases; do
    mkdir -p "$opt/manual_kernels/$case/tilelang/best"
    if [[ ! -e "$opt/manual_kernels/$case/tilelang/best/export" ]]; then
        ln -s "$opt/manual/$case/best/export" "$opt/manual_kernels/$case/tilelang/best/export"
    fi
    for round in 0 1; do
        if [[ "$round" == 0 ]]; then methods='original manual tileops'; else methods='tileops manual original'; fi
        for method in $methods; do
            output="$opt/compare_manual/round${round}/$method/$case"
            mkdir -p "$(dirname "$output")"
            if [[ "$method" == tileops ]]; then
                set +u; source envs/activate_gh200_latest.sh; set -u
                export PYTHONPATH=/u/yucheny8/argus:/u/yucheny8/argus/.tileops_work/src
                export PATH=/u/yucheny8/nsight-systems-2025.5/bin:$PATH
                unset TVM_LIBRARY_PATH TL_CUTLASS_PATH TL_TEMPLATE_PATH
                q=${case#mtpc_Q};q=${q%%_L*};length=${case##*_L}
                python -u -m scripts.bench_official_tileops_dense mtp "$length" --q-length "$q" --input-file "$root/nautilus/${case}_inputs.pt" --output "$output" --repeat 50 >"$output.log" 2>&1
            else
                source "$opt/env.sh"
                if [[ "$method" == original ]]; then export ARGUS_PAPER_KERNEL_ROOT=$root/kernels; else export ARGUS_PAPER_KERNEL_ROOT=$opt/manual_kernels; fi
                cp "$root/nautilus/${case}_inputs.pt" "$(dirname "$output")/${case}_inputs.pt"
                python -u -m scripts.bench_paper_sweep "$case" --output "$output" --repeat 50 >"$output.log" 2>&1
            fi
            echo "COMPARED $case round$round $method"
        done
    done
done
