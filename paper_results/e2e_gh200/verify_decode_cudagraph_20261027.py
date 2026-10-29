"""Validate full-step CUDA Graph capture, changed inputs/KV, and endpoint latency."""

import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import torch
import tvm_ffi
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
        name = f"paper_graph_L{length}"
        ALL_ATTENTION_FUNCTIONS.register(name, ALL_ATTENTION_FUNCTIONS["paper_e2e"])
        model.config._attn_implementation = name
        inputs = make_inputs()
        expected = model(**inputs).last_hidden_state.float().clone()
        assert len(cores) == 32
        for core in cores:
            core.reference_mode = False
        actual = model(**inputs).last_hidden_state.float()
        assert ((actual - expected).norm() / expected.norm()).item() < 0.01
        bundles.append({"length": length, "model": model, "inputs": inputs, "cores": cores,
                        "expected": expected, "description": description, "samples": []})
    assert bundles[0]["model"].paper_weights_sha256 == bundles[1]["model"].paper_weights_sha256
    for bundle in bundles:
        bundle["compiled"] = torch.compile(bundle["model"], dynamic=False, fullgraph=False)
    print("Warming both compiled models", flush=True)
    for _ in range(10):
        for bundle in bundles:
            bundle["compiled"](**bundle["inputs"])
    torch.cuda.synchronize()

    for bundle in bundles:
        length = bundle["length"]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with tvm_ffi.use_torch_stream(torch.cuda.stream(stream)):
            for _ in range(3):
                bundle["compiled"](**bundle["inputs"])
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        assert all(core.checked and not core.reference_mode for core in bundle["cores"])
        graph = torch.cuda.CUDAGraph()
        graph.enable_debug_mode()
        print(f"Capturing whole forward L{length}", flush=True)
        with tvm_ffi.use_torch_stream(torch.cuda.graph(graph, stream=stream)):
            assert tvm_ffi.get_raw_stream(tvm_ffi.device("cuda:0")) == stream.cuda_stream
            output = bundle["compiled"](**bundle["inputs"]).last_hidden_state
        graph.replay()
        torch.cuda.synchronize()
        error = ((output.float() - bundle["expected"]).norm()
                 / bundle["expected"].norm()).item()
        assert error < 0.01
        graph.debug_dump(str(directory / f"L{length}.dot"))
        token = bundle["inputs"]["inputs_embeds"]
        original_token = token.clone()
        correctness = []
        for case in range(2):
            torch.manual_seed(100 + case)
            token.normal_()
            expected = bundle["compiled"](**bundle["inputs"]).last_hidden_state.float().clone()
            graph.replay()
            torch.cuda.synchronize()
            actual = output.float()
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
            correctness.append({"case": f"changed_token_{case}",
                                "relative_l2": ((actual - expected).norm()
                                                / expected.norm()).item()})
        before_kv_change = output.float().clone()
        prefix_v = bundle["cores"][0].buffers[2][:, :, :-1]
        saved_prefix_v = prefix_v.clone()
        prefix_v.add_(0.5)
        expected = bundle["compiled"](**bundle["inputs"]).last_hidden_state.float().clone()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output.float(), expected, rtol=1e-3, atol=1e-3)
        changed = ((expected - before_kv_change).norm() / before_kv_change.norm()).item()
        assert changed > 1e-4
        correctness.append({"case": "changed_prefix_V", "output_change_relative_l2": changed,
                            "relative_l2": ((output.float() - expected).norm()
                                            / expected.norm()).item()})
        prefix_v.copy_(saved_prefix_v)
        token.copy_(original_token)
        graph.replay()
        torch.cuda.synchronize()
        restored_error = ((output.float() - bundle["expected"]).norm()
                          / bundle["expected"].norm()).item()
        assert restored_error < 0.01
        bundle.update(graph=graph, graph_output=output, capture_stream=stream,
                      correctness=correctness, relative_l2=restored_error)
        print(f"PASS L{length}: capture, changed tokens, changed prefix KV", flush=True)

    print("Warming replay", flush=True)
    for _ in range(10):
        for bundle in bundles:
            bundle["graph"].replay()
    torch.cuda.synchronize()
    print("Timing30 paired normal/replay rounds", flush=True)
    cases = [(bundle, mode) for bundle in bundles for mode in ("compiled", "graph")]
    for repeat in range(30):
        for bundle, mode in (cases if repeat % 2 == 0 else cases[::-1]):
            torch.cuda.synchronize()
            started = perf_counter()
            if mode == "graph":
                bundle["graph"].replay()
            else:
                bundle["compiled"](**bundle["inputs"])
            submitted = perf_counter()
            torch.cuda.synchronize()
            finished = perf_counter()
            bundle["samples"].append({"round": repeat, "mode": mode,
                                      "wall_ms": (finished - started) * 1000,
                                      "submit_ms": (submitted - started) * 1000})
    (directory / "timing_samples.json").write_text(json.dumps([
        {"kv_length": b["length"], "samples": b["samples"], "correctness": b["correctness"],
         "relative_l2": b["relative_l2"]} for b in bundles
    ], indent=2) + "\n")
    results = []
    for bundle in bundles:
        length = bundle["length"]
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        ) as profiler:
            for repeat in range(2):
                torch.cuda.synchronize()
                with torch.profiler.record_function(f"whole_graph_L{length}[{repeat}]"):
                    bundle["graph"].replay()
                    torch.cuda.synchronize()
        trace = directory / f"L{length}_graph.trace.json"
        profiler.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
        ranges = [e for e in events if e.get("cat") == "user_annotation"
                  and e.get("name", "").startswith(f"whole_graph_L{length}[")]
        assert len(ranges) == 2
        profiles = []
        for region in ranges:
            kernels = [e for e in events if e.get("cat") == "kernel"
                       and region["ts"] <= e["ts"]
                       and e["ts"] + e["dur"] <= region["ts"] + region["dur"]]
            attention = [e for e in kernels if e["name"].startswith("gqa_decode")]
            assert len(attention) == (32 if length == 256 else 64)
            assert all(e["args"].get("graph id", 0) != 0 for e in attention)
            profiles.append({"range": region["name"], "kernel_count": len(kernels),
                             "attention_kernel_count": len(attention),
                             "attention_graph_ids": sorted({e["args"]["graph id"]
                                                            for e in attention}),
                             "attention_gpu_ms": sum(e["dur"] for e in attention) / 1000,
                             "all_gpu_kernel_ms": sum(e["dur"] for e in kernels) / 1000})
        means = {mode: {key: statistics.mean(s[key] for s in bundle["samples"]
                                             if s["mode"] == mode)
                        for key in ("wall_ms", "submit_ms")} for mode in ("compiled", "graph")}
        result = {"kv_length": length, "means": means, "samples": bundle["samples"],
                  "correctness": bundle["correctness"], "relative_l2": bundle["relative_l2"],
                  "gpu": torch.cuda.get_device_name(),
                  "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                  "torch": torch.__version__, "baseline_source": bundle["cores"][0].source,
                  "profiles": profiles, "scope": "whole backbone single decode step; includes "
                  "new KV writes; fixed cache position; excludes token/prefix setup and capture"}
        results.append(result)
        print(length, means, profiles, flush=True)
    (directory / "results.json").write_text(json.dumps(results, indent=2) + "\n")
print("Finished whole-step CUDA Graph verification", flush=True)
