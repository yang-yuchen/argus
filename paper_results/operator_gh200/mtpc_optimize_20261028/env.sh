# Source after selecting case.
root=/u/yucheny8/argus/profiles_paper_kernels/mtpc_full_kv_20261028
opt=/u/yucheny8/argus/profiles_paper_kernels/mtpc_optimize_20261028
source /u/yucheny8/miniconda3/etc/profile.d/conda.sh
set +u
conda activate /u/yucheny8/miniconda3/envs/ntawa
set -u
export LD_PRELOAD=/usr/lib64/libstdc++.so.6
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PYTHONPATH=$root/source/upstream/python:$root/source/tilelang:/u/yucheny8/argus:$opt
export TVM_LIBRARY_PATH=/work/nvme/beqg/mbudiu2/neptune-release-build2
export TL_CUTLASS_PATH=/work/nvme/beqg/yucheny8/nautilus/3rdparty/tilelang/3rdparty/cutlass/include
export TL_TEMPLATE_PATH=$root/source/tilelang/src
export PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9/bin:$CONDA_PREFIX/bin:/u/yucheny8/nsight-systems-2025.5/bin:$PATH
export TMPDIR=$opt/tmp/$case TILELANG_CACHE_DIR=$opt/cache/tilelang/$case
export TILELANG_TMP_DIR=$TMPDIR/tilelang
export TRITON_CACHE_DIR=$opt/cache/triton/$case CUDA_CACHE_PATH=$opt/cache/cuda/$case
export ARGUS_PAPER_KERNEL_ROOT=$root/kernels
mkdir -p "$TMPDIR" "$TILELANG_TMP_DIR" "$TILELANG_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
ulimit -c 0
