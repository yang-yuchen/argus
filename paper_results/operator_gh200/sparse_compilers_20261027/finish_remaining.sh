#!/usr/bin/env bash
set -euo pipefail
cd /u/yucheny8/argus
python - <<'PY'
import json,shutil
from pathlib import Path
root=Path('profiles_sparse_compilers_20261027')
archive=root/'tilelang_defaults'
archive.mkdir(exist_ok=True)
for p in (root/'tilelang').glob('sparse_L*.json'):
    data=json.loads(p.read_text())
    if isinstance(data.get('source',{}).get('autotune'),dict):
        continue
    for extension in ['json','csv','log','nsys-rep','sqlite']:
        src=p.with_suffix('.'+extension)
        if src.exists():
            dst=archive/src.name
            assert not dst.exists(),dst
            shutil.move(src,dst)
PY
bash scripts/run_sparse_compiler_resume.sh
