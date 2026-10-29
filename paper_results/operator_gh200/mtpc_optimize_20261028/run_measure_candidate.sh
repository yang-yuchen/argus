#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
opt=/u/yucheny8/argus/profiles_paper_kernels/mtpc_optimize_20261028
case=${1:-$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$opt/manual_cases.txt")}
source "$opt/env.sh"
mkdir -p "$opt/manual_kernels/$case/tilelang/best" "$opt/measurement/manual"
if [[ ! -e "$opt/manual_kernels/$case/tilelang/best/export" ]]; then
    ln -s "$opt/manual/$case/best/export" "$opt/manual_kernels/$case/tilelang/best/export"
fi
export ARGUS_PAPER_KERNEL_ROOT=$opt/manual_kernels
cp "$root/nautilus/${case}_inputs.pt" "$opt/measurement/manual/${case}_inputs.pt"
python -u -m scripts.bench_paper_sweep "$case" --output "$opt/measurement/manual/$case" --repeat 50 >"$opt/measurement/manual/$case.log" 2>&1
echo "MEASURED OPTIMIZED CANDIDATE $case"
