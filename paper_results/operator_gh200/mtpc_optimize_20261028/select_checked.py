"""Choose only independently correct resumed-tuning candidates; retain the original fallback."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

import torch
import tvm
from tvm import meta_schedule as ms, tir
from tvm.target import Target
import neptune
from neptune.autotune import _kernel_ms, fresh_copy
from neptune.export import export_kernels
from argus.utils import is_unrecoverable_cuda_error
from manual_probe import ROOT, BASE, bind, checked, data_sets


def main(case):
    destination = ROOT / "retune" / case
    database_dir = destination / "tilelang/db"
    db = ms.database.JSONDatabase(str(database_dir/"database_workload.json"),
                                  str(database_dir/"database_tuning_record.json"), allow_missing=False)
    records = db.get_all_tuning_records()
    if not records:
        raise ValueError("No tuning records")
    records = db.get_top_k(records[0].workload, len(records))
    data = data_sets(case)
    target = Target("cuda")
    original_export = BASE / "kernels" / case / "tilelang/best/export"
    original_lib = tvm.runtime.load_module(str(original_export/"kernel.so"))
    original_run, original_checks = checked(original_lib, data)
    original_ms = statistics.mean([_kernel_ms(original_run), _kernel_ms(original_run)])
    results = [{"kind": "original", "correct": True, "screening_ms": original_ms,
                "validation": original_checks}]
    best = (original_ms, "original", None, original_export)
    valid_count = 0
    for index, record in enumerate(records):
        if valid_count >= 16:
            break
        costs = [float(x) for x in (record.run_secs or ())]
        if not costs or not all(math.isfinite(x) and 0 < x < 1e6 for x in costs):
            continue
        row = {"kind": "resumed_database", "rank": index, "correct": False,
               "tuner_ms": statistics.mean(costs)*1000}
        try:
            sch = tir.Schedule(fresh_copy(record.workload.mod), debug_mask=0)
            record.trace.apply_to_schedule(sch, remove_postproc=False)
            library = neptune.compile(sch.mod, target)
            call, checks = checked(library, data)
            times = [_kernel_ms(call), _kernel_ms(call)]
            measured_ms = statistics.mean(times)
            row.update(correct=True, screening_ms=measured_ms, validation=checks, times_ms=times)
            valid_count += 1
            if measured_ms < best[0]:
                best = (measured_ms, f"resumed_rank{index}", sch.mod, None)
        except Exception as error:
            if is_unrecoverable_cuda_error(error) or "CUDA error:" in str(error):
                raise
            row["error"] = str(error)[-2000:]
        results.append(row)
        (destination/"checked_candidates.json").write_text(json.dumps(results,indent=2)+"\n")
        print(case,row,flush=True)
    manual_summary = ROOT / "manual" / case / "summary.json"
    if manual_summary.exists():
        summary = json.loads(manual_summary.read_text())
        if summary.get("export"):
            manual_export = Path(summary["export"])
            manual_lib = tvm.runtime.load_module(str(manual_export/"kernel.so"))
            call, checks = checked(manual_lib, data)
            measured_ms = statistics.mean([_kernel_ms(call), _kernel_ms(call)])
            results.append({"kind":"manual", "correct":True, "screening_ms":measured_ms,
                            "validation":checks, "config":summary["best_config"]})
            if measured_ms < best[0]:
                best = (measured_ms, "manual", None, manual_export)
    selected = ROOT / "selected" / case / "tilelang/best/export"
    if selected.exists():
        raise ValueError(f"Refusing to overwrite {selected}")
    selected.parent.mkdir(parents=True,exist_ok=True)
    if best[2] is None:
        shutil.copytree(best[3],selected)
        library = tvm.runtime.load_module(str(selected/"kernel.so"))
    else:
        library,_ = export_kernels(best[2],target,str(selected),meta={
            "variant":case,"backend":"tilelang","origin":best[1],
            "attempt_budget":256,"independent_fp32_correctness":True})
    _,validation = checked(library,data)
    result = {"case":case,"selected_kind":best[1],"screening_ms":best[0],
              "original_screening_ms":original_ms,"correct_candidates_checked":valid_count,
              "validation":validation,"export":str(selected),
              "kernel_sha256":hashlib.sha256((selected/"kernel.so").read_bytes()).hexdigest(),
              "status":"candidate_selected; final counterbalanced nsys timing required"}
    (destination/"checked_candidates.json").write_text(json.dumps(results,indent=2)+"\n")
    (destination/"selection.json").write_text(json.dumps(result,indent=2)+"\n")
    print("STRICT SELECTION",result,flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("case")
    main(parser.parse_args().case)
