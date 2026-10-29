"""Parameter-only probes of the official Nautilus manual schedule."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "mtpc_full_kv_20261028"


def manual_configs(case):
    q = int(case.split("_Q")[1].split("_L")[0])
    length = int(case.split("_L")[1])
    if q % 16:
        return []
    if length == 256:
        pairs = [(1, 64), (2, 64), (4, 64), (2, 128), (4, 32), (8, 32)]
    else:
        pairs = list(itertools.product((16, 32, 64), (64, 128)))
    return [dict(num_splits=splits, tile=tile, num_warps=warps, num_stages=2,
                 tma=False, warp_specialize=None)
            for (splits, tile), warps in itertools.product(pairs, (4, 8))
            if length % (splits * tile) == 0]


def validate(call, output, reference):
    for poison in (float("nan"), 1.0) * 4:
        output.fill_(poison)
        call()
        actual = output.float()
        torch.testing.assert_close(actual, reference, rtol=1e-2, atol=3e-3)
        norm = reference.norm().item()
        difference = (actual - reference).norm().item()
        relative = difference / norm if norm else (0.0 if difference == 0 else float("inf"))
        if not bool(torch.isfinite(actual).all()) or relative >= 1e-2:
            raise AssertionError(f"Output is incomplete or relative L2={relative}")
    return {"max_abs_error": (actual-reference).abs().max().item(), "relative_l2": relative}


def data_sets(case):
    from scripts.bench_paper_sweep import reference
    torch.backends.cuda.matmul.allow_tf32 = False
    cpu = torch.load(BASE / "nautilus" / f"{case}_inputs.pt", map_location="cpu", weights_only=True)
    data = [[x.cuda().contiguous() for x in cpu]]
    generator = torch.Generator(device="cuda").manual_seed(17)
    data.append([torch.randn(x.shape, dtype=x.dtype, device="cuda", generator=generator) for x in cpu])
    return [(xs, reference(case, xs)) for xs in data]


def bind(library, tensors):
    import tvm
    output = torch.empty_like(tensors[0])
    args = [tvm.nd.from_dlpack(x) for x in (*tensors, output)]
    entry = library["main"]
    def run(_library=library):
        entry(*args)
    return run, output


def checked(library, data):
    checks = []
    first = None
    for tensors, reference in data:
        run, output = bind(library, tensors)
        checks.append(validate(run, output, reference))
        if first is None:
            first = run
    return first, checks


def run_case(case):
    import tvm
    import neptune
    import attn_variants as av
    from tvm import tir
    from tvm.target import Target
    from neptune.autotune import fresh_copy, _kernel_ms
    from neptune.export import export_kernels
    from neptune.operators.attn_plain import schedule_split_kv_flash
    from argus.utils import is_unrecoverable_cuda_error

    destination = ROOT / "manual" / case
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    data = data_sets(case)
    original_path = BASE / "kernels" / case / "tilelang/best/export/kernel.so"
    original_lib = tvm.runtime.load_module(str(original_path))
    original_run, original_checks = checked(original_lib, data)
    original_ms = _kernel_ms(original_run)
    target = Target("cuda")
    variant = av.get_variant(case)
    best = None
    for config in manual_configs(case):
        row = {"config": config, "correct": False}
        try:
            schedule = tir.Schedule(fresh_copy(variant.mod), debug_mask=0)
            schedule_split_kv_flash(schedule, **config)
            library = neptune.compile(schedule.mod, target)
            call, checks = checked(library, data)
            times = [_kernel_ms(call), _kernel_ms(call)]
            row.update(correct=True, validation=checks, times_ms=times, screening_ms=statistics.mean(times))
            if best is None or row["screening_ms"] < best[0]:
                best = (row["screening_ms"], schedule.mod, config)
        except Exception as error:
            if is_unrecoverable_cuda_error(error) or "CUDA error:" in str(error):
                raise
            row["error"] = str(error)[-1800:]
        results.append(row)
        (destination / "candidates.json").write_text(json.dumps(results, indent=2)+"\n")
        print(case, row, flush=True)
    summary = {"case": case, "original_screening_ms": original_ms,
               "original_validation": original_checks, "candidate_count": len(results),
               "valid_candidates": sum(r["correct"] for r in results)}
    if best is not None:
        export = destination / "best/export"
        exported, _ = export_kernels(best[1], target, str(export), meta={
            "variant": case, "backend": "tilelang", "origin": "upstream manual schedule parameters",
            "configuration": best[2]})
        _, checks = checked(exported, data)
        summary.update(best_config=best[2], best_screening_ms=best[0],
                       validation=checks, export=str(export),
                       kernel_sha256=hashlib.sha256((export/"kernel.so").read_bytes()).hexdigest())
    (destination/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("MANUAL SUMMARY",summary,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    args = parser.parse_args()
    run_case(args.case)
