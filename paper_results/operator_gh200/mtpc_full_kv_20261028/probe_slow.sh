#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
root=profiles_paper_kernels/mtpc_full_kv_20261028/probes_slow_20261028
mkdir -p "$root"
set +u
source /u/yucheny8/miniconda3/etc/profile.d/conda.sh
conda activate /u/yucheny8/miniconda3/envs/ntawa
set -u
export LD_PRELOAD=/usr/lib64/libstdc++.so.6
export PYTHONPATH=/u/yucheny8/argus/profiles_paper_kernels/mtpc_full_kv_20261028/source/upstream/python:/u/yucheny8/argus/profiles_paper_kernels/mtpc_full_kv_20261028/source/tilelang:/u/yucheny8/argus
export TVM_LIBRARY_PATH=/work/nvme/beqg/mbudiu2/neptune-release-build2
export ARGUS_PAPER_KERNEL_ROOT=/u/yucheny8/argus/profiles_paper_kernels/mtpc_full_kv_20261028/kernels
export PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9/bin:$CONDA_PREFIX/bin:/u/yucheny8/nsight-systems-2025.5/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
for case in mtpc_Q4_L32768 mtpc_Q32_L32768 mtpc_Q64_L256 mtpc_Q64_L32768; do
  q=${case#mtpc_Q}; q=${q%%_L*}; l=${case##*_L}
  python -u -m scripts.bench_paper_sweep "$case" --output "$root/nautilus_$case" --repeat 100 >"$root/nautilus_$case.log" 2>&1
done
set +u; source envs/activate_gh200_latest.sh; set -u
export PYTHONPATH=/u/yucheny8/argus:/u/yucheny8/argus/.tileops_work/src
export PATH=/u/yucheny8/nsight-systems-2025.5/bin:$PATH
unset TVM_LIBRARY_PATH
for case in mtpc_Q4_L32768 mtpc_Q32_L32768 mtpc_Q64_L256 mtpc_Q64_L32768; do
  q=${case#mtpc_Q}; q=${q%%_L*}; l=${case##*_L}
  python -u -m scripts.bench_official_tileops_dense mtp "$l" --q-length "$q" --input-file "profiles_paper_kernels/mtpc_full_kv_20261028/nautilus/${case}_inputs.pt" --output "$root/tileops_$case" --repeat 100 >"$root/tileops_$case.log" 2>&1
done
