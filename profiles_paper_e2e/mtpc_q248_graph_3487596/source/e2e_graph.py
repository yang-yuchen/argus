"""Whole-step graph capture for fixed-address decode and MTP benchmarks."""

from contextlib import contextmanager
import json
import math
import statistics

import torch


@contextmanager
def backend_stream(implementation, torch_context):
    """Align the selected runtime with Torch's warmup/capture stream."""
    if implementation in ("tileops_bound", "tileops"):
        import tvm_ffi

        with tvm_ffi.use_torch_stream(torch_context):
            yield
    elif implementation == "nautilus":
        import tvm

        device = tvm.cuda(torch.cuda.current_device())
        outer = torch.cuda.current_stream().cuda_stream
        # Legacy TVM has no stream getter; this harness establishes the Torch-stream invariant.
        device.set_raw_stream(outer)
        try:
            with torch_context:
                device.set_raw_stream(torch.cuda.current_stream().cuda_stream)
                yield
        finally:
            device.set_raw_stream(outer)
    elif implementation == "flex":
        with torch_context:
            yield
    else:
        raise ValueError(f"Unsupported graph backend: {implementation}")


def capture_decode_graph(compiled, inputs, cores, reference, implementation="tileops_bound",
                         variant="decode"):
    """Capture once, then validate changed token/KV values before returning replay."""
    if variant not in ("decode", "mtpc"):
        raise ValueError(f"Unsupported graph variant: {variant}")
    expected_cores = 256 if variant == "mtpc" else 32
    if len(cores) != expected_cores or not all(c.checked and not c.reference_mode for c in cores):
        raise ValueError(f"Graph capture requires{expected_cores} validated attention cores")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with backend_stream(implementation, torch.cuda.stream(stream)):
        for _ in range(3):
            compiled(**inputs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with backend_stream(implementation, torch.cuda.graph(graph, stream=stream)):
        output = compiled(**inputs).last_hidden_state

    def replay(_compiled=compiled, _inputs=inputs, _cores=cores, _stream=stream):
        # Keep every custom-launch buffer and its owner alive for the graph lifetime.
        graph.replay()
        return output

    token = inputs["inputs_embeds"]
    original_token = token.clone()

    def check_changed_input():
        expected = compiled(**inputs).last_hidden_state.float().clone()
        # Poison fixed Q and any persistent backend output to rule out stale-output success.
        for core in cores:
            core.buffers[0].fill_(float("nan"))
            core.call().fill_(float("nan"))
        actual = replay().float()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        error = ((actual - expected).norm() / expected.norm().clamp_min(1e-30)).item()
        if not math.isfinite(error) or error >= 0.01:
            raise AssertionError(f"Graph output relativeL2={error}")
        return error

    changed_errors = []
    for case in range(2):
        torch.manual_seed(100 + case)
        token.normal_()
        changed_errors.append(check_changed_input())
    before = output.float().clone()
    query_length = cores[0].buffers[0].shape[2]
    prefix_v = cores[0].buffers[2][:, :, :-query_length]
    saved_v = prefix_v.clone()
    prefix_v.add_(0.5)
    prefix_error = check_changed_input()
    change = ((output.float() - before).norm() / before.norm().clamp_min(1e-30)).item()
    if not math.isfinite(change) or change <= 1e-4:
        raise AssertionError("Graph output did not respond to changed historical KV")
    prefix_v.copy_(saved_v)
    token.copy_(original_token)
    actual = replay().float()
    torch.cuda.synchronize()
    restored_error = ((actual - reference).norm() / reference.norm().clamp_min(1e-30)).item()
    if not math.isfinite(restored_error) or restored_error >= 0.01:
        raise AssertionError(f"Restored graph relativeL2={restored_error}")
    return replay, {
        "changed_token_relative_l2": changed_errors,
        "changed_prefix_v_relative_l2": prefix_error,
        "prefix_v_output_change_relative_l2": change,
        "restored_relative_l2": restored_error,
        "shared_buffers_poisoned_before_replay": True,
        "attention_group_count": expected_cores,
    }


def inspect_graph_trace(events, length, implementation="tileops_bound", variant="decode"):
    """Require two complete replay ranges, all GPU kernels captured, and per-layer KV writes."""
    ranges = [e for e in events if e.get("cat") == "user_annotation"
              and e.get("name", "").startswith("paper_graph_audit[")]
    if len(ranges) != 2:
        raise ValueError("Expected2 graph replay audit ranges")
    profiles = []
    for region in ranges:
        kernels = [e for e in events if e.get("cat") == "kernel"
                   and region["ts"] <= e["ts"]
                   and e["ts"] + e["dur"] <= region["ts"] + region["dur"]]
        if variant == "mtpc":
            if implementation == "tileops":
                attention = [e for e in kernels if e["name"] in ("parts_kernel", "main_kernel")]
                stages = {name: sum(e["name"] == name for e in attention)
                          for name in ("main_kernel", "parts_kernel")}
                expected_stages = {"main_kernel": 256,
                                   "parts_kernel": 0 if length <= 256 else 256}
                if stages != expected_stages:
                    raise ValueError("Missing captured MTP attention stage")
            elif implementation == "nautilus":
                attention = [e for e in kernels if e["name"].startswith("main_kernel")]
                if any(sum(e["name"] == name for e in attention) != 256
                       for name in ("main_kernel_kernel", "main_kernel_1_kernel")):
                    raise ValueError("Missing captured MTP attention stage")
            elif implementation == "flex":
                attention = [e for e in kernels if "flex_attention" in e["name"]
                             or "flex_decoding" in e["name"]]
            else:
                raise ValueError(f"Unsupported MTP graph backend: {implementation}")
        elif variant != "decode":
            raise ValueError(f"Unsupported graph variant: {variant}")
        elif implementation == "tileops_bound":
            attention = [e for e in kernels if e["name"].startswith("gqa_decode")]
            expected_attention = 32 if length < 640 else 64
        elif implementation == "nautilus":
            attention = [e for e in kernels if e["name"].startswith("main_kernel")]
            expected_attention = 64
        elif implementation == "flex":
            attention = [e for e in kernels if "flex_attention" in e["name"]
                         or "flex_decoding" in e["name"]]
            expected_attention = 96  # Decode template plus two official split-KV reductions.
        else:
            raise ValueError(f"Unsupported graph backend: {implementation}")
        kv_updates = [e for e in kernels if "index_copy" in e["name"]]
        if variant == "mtpc":
            expected_mtp = (256 if implementation == "tileops" and length <= 256 else
                            768 if implementation == "flex" else 512)
            if len(attention) != expected_mtp:
                raise ValueError("Missing captured MTP attention groups")
        elif len(attention) != expected_attention:
            raise ValueError("Missing captured attention or KV-update kernels")
        if len(kv_updates) != 64:
            raise ValueError("Missing captured attention or KV-update kernels")
        if not all(e["args"].get("graph id", 0) > 0 for e in kernels):
            raise ValueError("GPU kernel executed outside the captured graph")
        profiles.append({"kernel_count": len(kernels), "attention_kernel_count": len(attention),
                         "kv_update_kernel_count": len(kv_updates), "all_kernels_in_graph": True,
                         "attention_gpu_ms": sum(e["dur"] for e in attention) / 1000,
                         "all_gpu_kernel_ms": sum(e["dur"] for e in kernels) / 1000})
    if any(profiles[0][key] != profiles[1][key] for key in (
        "kernel_count", "attention_kernel_count", "kv_update_kernel_count",
    )):
        raise ValueError("Graph replay kernel counts changed between forwards")
    result = profiles[0].copy()
    for key in ("attention_gpu_ms", "all_gpu_kernel_ms"):
        result[key] = statistics.mean(p[key] for p in profiles)
    return result


def audit_graph_replay(replay, length, trace_path, implementation="tileops_bound", variant="decode"):
    """Profile separately from formal wall samples and save the completeness evidence."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    ) as profiler:
        for repeat in range(2):
            torch.cuda.synchronize()
            with torch.profiler.record_function(f"paper_graph_audit[{repeat}]"):
                replay()
                torch.cuda.synchronize()
    profiler.export_chrome_trace(str(trace_path))
    return inspect_graph_trace(json.loads(trace_path.read_text())["traceEvents"], length,
                               implementation, variant)
