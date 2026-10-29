#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
source envs/activate_gh200_latest.sh
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/u/yucheny8/argus/.paper_e2e_hf:/u/yucheny8/argus:/u/yucheny8/argus/.tileops_work/src
unset TVM_LIBRARY_PATH
python -u profiles_paper_e2e/diagnose_decode_timing_20261027.py \
    "profiles_paper_e2e/decode_timing_audit_20261027_j${SLURM_JOB_ID:?}"
