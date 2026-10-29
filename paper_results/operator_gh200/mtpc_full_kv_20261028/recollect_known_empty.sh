#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
root=/u/yucheny8/argus/profiles_paper_kernels/mtpc_full_kv_20261028
for case in mtpc_Q2_L512 mtpc_Q2_L32768 mtpc_Q4_L512 mtpc_Q4_L8192; do
    FORCE_REMEASURE=1 bash "$root/run_missing_case.sh" "$case"
done
