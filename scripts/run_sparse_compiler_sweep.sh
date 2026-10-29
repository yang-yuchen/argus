#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
export CASES='sparse_L256 sparse_L512 sparse_L1024 sparse_L2048 sparse_L4096 sparse_L8192 sparse_L16384 sparse_L32768'
bash scripts/run_paper_kernels_sweep.sh profiles_sparse_compilers_20261027/nautilus
bash scripts/run_sparse_official_baselines.sh
