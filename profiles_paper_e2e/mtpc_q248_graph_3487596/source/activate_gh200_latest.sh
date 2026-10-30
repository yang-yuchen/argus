#!/usr/bin/env bash
# Source this file to activate the 2026-10-24 GH200 baseline environment.
source /u/yucheny8/miniconda3/etc/profile.d/conda.sh
conda activate /work/nvme/beqg/yucheny8/envs/argus-gh200-latest || return

export ARGUS_GH200_WORK=/work/nvme/beqg/yucheny8/argus-gh200-latest
export CUDA_HOME="${CONDA_PREFIX}/lib/python3.13/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
# NVHPC modules otherwise inject CUDA 12.9 headers ahead of pip's CUDA 13 headers.
export CPATH="$CUDA_HOME/include:$CUDA_HOME/include/cccl"
unset CPLUS_INCLUDE_PATH C_INCLUDE_PATH
ln -sf "$CUDA_HOME/lib/libcudart.so.13" "$CUDA_HOME/lib/libcudart.so"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14
export PYTHONNOUSERSITE=1
export CONDA_PKGS_DIRS=/work/nvme/beqg/yucheny8/conda-pkgs
export PIP_CACHE_DIR=/work/nvme/beqg/yucheny8/pip-cache
export XDG_CACHE_HOME="$ARGUS_GH200_WORK/cache"
export TMPDIR="$ARGUS_GH200_WORK/tmp"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TILELANG_CACHE_DIR="$XDG_CACHE_HOME/tilelang"
export TILELANG_TMP_DIR="$TMPDIR/tilelang"
export HELION_CACHE_DIR="$XDG_CACHE_HOME/helion"
export FLASHINFER_WORKSPACE_BASE="$ARGUS_GH200_WORK"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MAX_JOBS=8
export FLASH_ATTN_CUDA_ARCHS=90

mkdir -p "$TMPDIR" "$TILELANG_TMP_DIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
# A failed CUDA compiler or kernel must not leave a multi-GB core in the repo.
ulimit -c 0
