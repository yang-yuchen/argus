#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
source /u/yucheny8/miniconda3/etc/profile.d/conda.sh
set +u; conda activate /u/yucheny8/miniconda3/envs/ntawa; set -u
export LD_PRELOAD=/usr/lib64/libstdc++.so.6
export PYTHONPATH=/u/yucheny8/argus/profiles_paper_kernels/mtpc_full_kv_20261028/source/upstream/python:/u/yucheny8/argus/profiles_paper_kernels/mtpc_full_kv_20261028/source/tilelang:/u/yucheny8/argus
export TVM_LIBRARY_PATH=/work/nvme/beqg/mbudiu2/neptune-release-build2
export ARGUS_PAPER_KERNEL_ROOT=/u/yucheny8/argus/profiles_paper_kernels/mtpc_full_kv_20261028/kernels
python profiles_paper_kernels/mtpc_full_kv_20261028/diagnose_q64_l8192.py
