#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
output=profiles_paper_e2e/mtpc_ghx4_20261027_j3470074
export PYTHONPATH=/work/nvme/beqg/yucheny8/argus-paper-kernels/runtime_py313:/u/yucheny8/argus
export TVM_LIBRARY_PATH=/work/nvme/beqg/yucheny8/src/neptune-internal-refactor/build/temp.linux-aarch64-cpython-311
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 USE_TF=0 USE_FLAX=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export TRITON_CACHE_DIR=/tmp/argus-paper-e2e-triton
export CUDA_CACHE_PATH=/tmp/argus-paper-e2e-cuda
nt_lib=/u/yucheny8/miniconda3/envs/nt/lib
nvidia_lib="$nt_lib/python3.13/site-packages/nvidia"
export LD_LIBRARY_PATH="$nvidia_lib/cublas/lib:$nvidia_lib/cuda_runtime/lib:$nt_lib:${LD_LIBRARY_PATH:-}"
ulimit -c 0
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
/u/yucheny8/miniconda3/envs/nt/bin/python -u -m scripts.bench_paper_e2e mtpc \
  --kv-length 65536 --implementation nautilus --repeat 20 --warmup 3 \
  --output "$output/mtpc_L65536" >"$output/mtpc_L65536.log" 2>&1
