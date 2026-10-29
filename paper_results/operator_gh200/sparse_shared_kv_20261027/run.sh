#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
export SPARSE_OUTPUT_ROOT=profiles_sparse_shared_kv_20261027
bash scripts/run_sparse_compiler_resume.sh flex helion
/work/nvme/beqg/yucheny8/envs/argus-gh200-latest/bin/python -m scripts.summarize_sparse_compilers "$SPARSE_OUTPUT_ROOT"
/work/nvme/beqg/yucheny8/envs/argus-gh200-latest/bin/python - <<'PY'
import json
from pathlib import Path
from scripts.summarize_sparse_compilers import LENGTHS, validated

root = Path('profiles_sparse_shared_kv_20261027')
for length in LENGTHS:
    case = f'sparse_L{length}.json'
    original = validated(root / 'nautilus' / case)
    for method in ('flex', 'helion'):
        data = validated(root / method / case, inputs_hash=original['inputs_sha256'])
        assert data['source']['shared_mla_kv'] is True, (method, length)
manifest_path = root / 'experiment_manifest.json'
manifest = json.loads(manifest_path.read_text())
manifest.update(status='measurements_validated', validated_rerun_points=16)
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
print('All 16 shared-KV rerun points validated.')
PY
/work/nvme/beqg/yucheny8/envs/argus-gh200-latest/bin/python plot_from_csv.py \
    --input-fp16 "$SPARSE_OUTPUT_ROOT/plot_input.csv" --operators SPARSE \
    --suffix=-bf16-tc1-sparse-d35a206-shared-kv --output-dir plots_from_csv
