#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
bash scripts/run_official_tilelang_decode.sh profiles_official_tilelang_decode_autotuned_20261026
bash scripts/run_official_tilelang_mtpc.sh profiles_official_tilelang_mtpc_autotuned_20261026
