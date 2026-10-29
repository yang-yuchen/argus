"""Pair short/long decode wall timings and separately profile GPU work and idle gaps."""

import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from scripts.bench_paper_e2e import AttentionCore, build_decode_model


directory = Path(sys.argv[1])
directory.mkdir(parents=True, exist_ok=False)
torch._dynamo.config.recompile_limit = 300
bundles = []
with torch.inference_mode():
    for length in (256, 32768):
        print(f"Building L{length}", flush=True)
        torch.manual_seed(0)
        cores = []

        def factory(variant, inputs, persistent_kv=False, kv_length=length, bound_cores=cores):
            core = AttentionCore(variant, inputs, "tileops_bound", kv_length, persistent_kv)
            bound_cores.append(core)
            return core

        model, make_inputs, description = build_decode_model(
            factory, variant="decode", kv_length=length, implementation="tileops_bound",
        )
        # Give each model its own registered hook/cache before compiling either model.
        name = f"paper_timing_L{length}"
        ALL_ATTENTION_FUNCTIONS.register(name, ALL_ATTENTION_FUNCTIONS["paper_e2e"])
        model.config._attn_implementation = name
        inputs = make_inputs()
        expected = model(**inputs).last_hidden_state.float().clone()
        assert len(cores) == 32
        for core in cores:
            core.reference_mode = False
        actual = model(**inputs).last_hidden_state.float()
        error = ((actual - expected).norm() / expected.norm()).item()
        assert error < 0.01
        bundles.append({"length": length, "model": model, "inputs": inputs, "cores": cores,
                        "expected": expected, "description": description, "relative_l2": error,
                        "samples": [], "begin": torch.cuda.Event(enable_timing=True),
                        "end": torch.cuda.Event(enable_timing=True)})
    assert bundles[0]["model"].paper_weights_sha256 == bundles[1]["model"].paper_weights_sha256
    for bundle in bundles:
        bundle["compiled"] = torch.compile(bundle["model"], dynamic=False, fullgraph=False)
    print("Alternating warmup", flush=True)
    for _ in range(10):
        for bundle in bundles:
            actual = bundle["compiled"](**bundle["inputs"]).last_hidden_state.float()
    torch.cuda.synchronize()
    for bundle in bundles:
        actual = bundle["compiled"](**bundle["inputs"]).last_hidden_state.float()
        error = ((actual - bundle["expected"]).norm() / bundle["expected"].norm()).item()
        assert error < 0.01
        bundle["compiled_relative_l2"] = error

    print("Measuring30 alternating rounds; order reversed every round", flush=True)
    for repeat in range(30):
        for bundle in (bundles if repeat % 2 == 0 else bundles[::-1]):
            torch.cuda.synchronize()
            bundle["begin"].record()
            started = perf_counter()
            bundle["compiled"](**bundle["inputs"])
            submitted = perf_counter()
            bundle["end"].record()
            torch.cuda.synchronize()
            finished = perf_counter()
            bundle["samples"].append({"round": repeat, "wall_ms": (finished - started) * 1000,
                                      "submit_ms": (submitted - started) * 1000,
                                      "event_span_ms": bundle["begin"].elapsed_time(bundle["end"])})

    results = []
    for bundle in bundles:
        length = bundle["length"]
        print(f"Profiling2 separate L{length} forwards", flush=True)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as profiler:
            for repeat in range(2):
                torch.cuda.synchronize()
                with torch.profiler.record_function(f"timing_audit_L{length}[{repeat}]"):
                    bundle["compiled"](**bundle["inputs"])
                    torch.cuda.synchronize()
        trace_path = directory / f"L{length}.trace.json"
        profiler.export_chrome_trace(str(trace_path))
        events = json.loads(trace_path.read_text())["traceEvents"]
        ranges = [event for event in events if event.get("cat") == "user_annotation"
                  and event.get("name", "").startswith(f"timing_audit_L{length}[")]
        assert len(ranges) == 2
        profiles = []
        for region in ranges:
            gpu = [event for event in events
                   if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
                   and region["ts"] <= event["ts"]
                   and event["ts"] + event["dur"] <= region["ts"] + region["dur"]]
            kernels = [event for event in gpu if event["cat"] == "kernel"]
            attention = [event for event in kernels if event["name"].startswith("gqa_decode")]
            assert len(attention) == (32 if length == 256 else 64)
            intervals = sorted((event["ts"], event["ts"] + event["dur"]) for event in gpu)
            busy, end = 0.0, float("-inf")
            for start, stop in intervals:
                busy += max(0.0, stop - max(start, end))
                end = max(end, stop)
            span = max(stop for _, stop in intervals) - intervals[0][0]
            profiles.append({"range": region["name"], "profiled_wall_ms": region["dur"] / 1000,
                             "gpu_kernel_sum_ms": sum(e["dur"] for e in kernels) / 1000,
                             "attention_kernel_sum_ms": sum(e["dur"] for e in attention) / 1000,
                             "gpu_busy_union_ms": busy / 1000, "gpu_span_ms": span / 1000,
                             "gpu_idle_within_span_ms": (span - busy) / 1000,
                             "kernel_count": len(kernels),
                             "streams": sorted({e["args"]["stream"] for e in kernels})})
        result = {"kv_length": length, "description": bundle["description"],
                  "gpu": torch.cuda.get_device_name(),
                  "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                  "torch": torch.__version__, "warmup_rounds": 10, "samples": bundle["samples"],
                  "means": {key: statistics.mean(s[key] for s in bundle["samples"])
                            for key in ("wall_ms", "submit_ms", "event_span_ms")},
                  "baseline_source": bundle["cores"][0].source,
                  "relative_l2": bundle["relative_l2"],
                  "compiled_relative_l2": bundle["compiled_relative_l2"], "profiles": profiles,
                  "event_note": "CUDA event span includes GPU idle; kernel sum comes from "
                                "separate profiled forwards, never substituted for wall samples"}
        results.append(result)
        print(length, result["means"], profiles, flush=True)
    (directory / "results.json").write_text(json.dumps(results, indent=2) + "\n")
print("Finished paired timing audit", flush=True)
