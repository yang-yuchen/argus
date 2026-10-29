"""FP64/RMSE equivalence check for the current PDF prefill baselines."""
import json
from pathlib import Path
import traceback
import torch
from argus import operators as ops
from scripts.error import naive_attention_fp64, compute_rmse
from scripts.bench_gh200_latest import runners

inputs = torch.load(Path('profiles_paper_kernels/run_20261025_02/prefill_inputs.pt'), weights_only=True)
inputs = [x.cuda().contiguous() for x in inputs]
reference = naive_attention_fp64(*inputs, is_causal=False, chunk_size=128)
results = {}
for runner in runners(ops.PF_GLOBAL, causal=False, gqa=False, flex_max_autotune=True):
    name = f'{runner.provider_name}({runner.impl_name})' if runner.impl_name else runner.provider_name
    row = {'name': name}
    try:
        if not runner.setup(*inputs):
            row['status'] = 'unsupported'
        else:
            runner.run(); torch.cuda.synchronize()
            output = runner.get_output().to(torch.float64)
            diff = output - reference
            row.update(status='passed', rmse=compute_rmse(reference, output),
                       max_abs=diff.abs().max().item(),
                       relative_l2=(diff.norm() / reference.norm()).item(),
                       finite=bool(torch.isfinite(output).all()))
    except Exception:
        row.update(status='failed', error=traceback.format_exc())
    results[name] = row
    try: runner.free_members_memory()
    except Exception: pass
Path('profiles_pdf_latest/accuracy.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
