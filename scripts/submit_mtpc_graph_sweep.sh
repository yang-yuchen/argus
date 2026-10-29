#!/usr/bin/env bash
set -euo pipefail
exec bash /u/yucheny8/argus/scripts/run_mtpc_graph_sweep_ghx4.sh \
    "/u/yucheny8/argus/profiles_paper_e2e/mtpc_graph_sweep_${SLURM_JOB_ID}"
