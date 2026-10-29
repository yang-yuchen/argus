"""Adjust only sampled KV tiling decisions of a published same-Q Nautilus trace."""
import hashlib
import json
from pathlib import Path
import statistics

import tvm
from tvm import tir
from tvm.target import Target
from tvm.tir.schedule.trace import _json_from_tvm
import neptune
import attn_variants as av
from neptune.autotune import fresh_copy, _kernel_ms
from neptune.export import export_kernels
from manual_probe import ROOT, BASE, checked, data_sets

case="mtpc_Q4_L32768"
destination=ROOT/"manual"/case
destination.mkdir(parents=True,exist_ok=True)
donor=Path("/work/nvme/beqg/mbudiu2/paper_kernels_sweep/L65536/mtpc_Q4_L65536/tilelang_w/best/trace.py")
text=donor.read_text()
assert text.count("decision=[1024, 64]")==1
assert text.count("decision=[64, 16]")==1
(destination/"donor_trace.py").write_text(text)
data=data_sets(case)
original=tvm.runtime.load_module(str(BASE/"kernels"/case/"tilelang/best/export/kernel.so"))
original_run,original_checks=checked(original,data)
original_ms=_kernel_ms(original_run)
variant=av.get_variant(case)
target=Target("cuda")
rows=[];best=None
for splits,rolling_tiles in ((32,16),(64,8)):
    config={"kv_tile":64,"kv_outer_tiles":512,"num_splits":splits,"rolling_tiles":rolling_tiles,
            "query_padding":16,"donor_case":"mtpc_Q4_L65536"}
    row={"config":config,"correct":False}
    try:
        schedule=tir.Schedule(fresh_copy(variant.mod),debug_mask=0)
        trace_data=_json_from_tvm(tvm.ir.load_json(donor.with_suffix('.json').read_text()))
        assert [int(x) for x in trace_data[1][0][1]] == [1024,64]
        assert [int(x) for x in trace_data[1][2][1]] == [64,16]
        trace_data[1][0][1]=[512,64]
        trace_data[1][2][1]=[splits,rolling_tiles]
        tir.schedule.Trace.apply_json_to_schedule(trace_data,schedule)
        library=neptune.compile(schedule.mod,target)
        run,checks=checked(library,data)
        times=[_kernel_ms(run),_kernel_ms(run)]
        score=statistics.mean(times)
        row.update(correct=True,validation=checks,times_ms=times,screening_ms=score)
        if best is None or score<best[0]:best=(score,schedule.mod,config)
    except Exception as error:
        if "CUDA error:" in str(error):raise
        row["error"]=str(error)[-3000:]
    rows.append(row)
    print(case,row,flush=True)
    (destination/"candidates.json").write_text(json.dumps(rows,indent=2)+"\n")
if best is None:raise RuntimeError("No independently correct donor-trace candidate")
export=destination/"best/export"
library,_=export_kernels(best[1],target,str(export),meta={"variant":case,"backend":"tilelang",
    "origin":"published same-Q trace; only KV sampling parameters adjusted","configuration":best[2]})
_,validation=checked(library,data)
(destination/"trace_recipe.json").write_text(json.dumps({
    "donor":str(donor.with_suffix('.json')),
    "donor_sha256":hashlib.sha256(donor.with_suffix('.json').read_bytes()).hexdigest(),
    "parameters":best[2],"replay_api":"tir.schedule.Trace.apply_json_to_schedule"},indent=2)+"\n")
result={"case":case,"original_screening_ms":original_ms,"original_validation":original_checks,
        "candidate_count":len(rows),"valid_candidates":sum(r["correct"] for r in rows),
        "best_config":best[2],"best_screening_ms":best[0],"validation":validation,
        "export":str(export),"kernel_sha256":hashlib.sha256((export/"kernel.so").read_bytes()).hexdigest()}
(destination/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
print("TRACE PARAMETER SUMMARY",result,flush=True)

# Reuse the allocated GH200 to measure only this optimized export.
import os
import shutil
import subprocess
import sys
manual_root = ROOT / "manual_kernels"
link = manual_root / case / "tilelang/best/export"
link.parent.mkdir(parents=True, exist_ok=True)
if not link.exists():
    link.symlink_to(export)
measurement = ROOT / "measurement/manual"
measurement.mkdir(parents=True, exist_ok=True)
shutil.copyfile(BASE / "nautilus" / f"{case}_inputs.pt", measurement / f"{case}_inputs.pt")
with (measurement / f"{case}.log").open("w") as log:
    subprocess.run([sys.executable, "-u", "-m", "scripts.bench_paper_sweep", case,
                    "--output", str(measurement / case), "--repeat", "50"],
                   env={**os.environ, "ARGUS_PAPER_KERNEL_ROOT": str(manual_root)},
                   stdout=log, stderr=subprocess.STDOUT, check=True)
print("MEASURED Q4 OPTIMIZED CANDIDATE", flush=True)
