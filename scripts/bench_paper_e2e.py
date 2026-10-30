"""Model forward latency for fixed decode/MTP/sparse paper kernels (no downloads)."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

import torch
from torch.nn import functional as F

from argus.paper_baselines.workloads import build_exported, export_path, metadata, reference


VARIANTS = ("decode", "mtpc", "sparse", "sparse_split")


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
        digest.update(memoryview(raw))
    return digest.hexdigest()


class AttentionCore:
    """Bind permanent KV addresses and prepare only Q for native-cache decode/MTP."""

    def __init__(self, variant, inputs, implementation, kv_length=None, persistent_kv=False,
                 query_length=16):
        self.variant = variant
        self.persistent_kv = persistent_kv
        self.implementation = "tilelang" if implementation == "baseline" else implementation
        self.reference_mode = True
        self.checked = False
        if persistent_kv:
            q = inputs[0]
            if self.implementation in ("tileops", "tileops_bound", "tilelang", "baseline"):
                q = torch.empty_like(q.transpose(1, 2),
                                     memory_format=torch.contiguous_format).transpose(1, 2)
            else:
                q = torch.empty_like(q, memory_format=torch.contiguous_format)
            self.buffers = [q, *inputs[1:]]
        elif variant in ("decode", "mtpc") and self.implementation in ("tileops", "tileops_bound"):
            # Stage straight into TileOPs' native layout instead of making two KV copies.
            self.buffers = [torch.empty_like(x.transpose(1, 2),
                                            memory_format=torch.contiguous_format).transpose(1, 2)
                            for x in inputs]
        else:
            self.buffers = [torch.empty_like(x, memory_format=torch.contiguous_format)
                            for x in inputs]
        for i, (dst, src) in enumerate(zip(self.buffers, inputs)):
            if i == 0 or not persistent_kv:
                dst.copy_(src)
        self.source = None
        if self.implementation == "nautilus":
            options = {"kv_length": kv_length} if kv_length is not None else {}
            if query_length != 16:
                options["query_length"] = query_length
            self.call = build_exported(variant, self.buffers, **options)
        elif variant in ("decode", "mtpc") and self.implementation in (
            "tileops", "tileops_bound", "flex",
        ):
            from argus.paper_baselines.e2e_mtp_baselines import build_mtp_baseline

            self.call, self.source = build_mtp_baseline(self.implementation, self.buffers)
        elif self.implementation == "tilelang" and variant == "decode":
            from argus.paper_baselines.dense import build_dense_baseline

            self.call = build_dense_baseline(variant, self.buffers, refresh_inputs=True)
        elif self.implementation == "tilelang" and variant.startswith("sparse"):
            from argus.paper_baselines.sparse import build_sparse_baseline

            self.call = build_sparse_baseline(variant, self.buffers)
        elif self.implementation == "sdpa" and variant == "decode":
            self.call = lambda: F.scaled_dot_product_attention(*self.buffers)
        else:
            raise ValueError(f"Unsupported backend {implementation} for {variant}")

    @torch.compiler.disable
    def __call__(self, *inputs):
        if len(inputs) != len(self.buffers):
            raise ValueError("Model attention input count changed")
        for i, (dst, src) in enumerate(zip(self.buffers, inputs)):
            if src.shape != dst.shape or src.dtype != dst.dtype or src.device != dst.device:
                raise ValueError("Model attention shape/dtype/device changed")
            if i > 0 and self.persistent_kv:
                if src.data_ptr() != dst.data_ptr() or src.stride() != dst.stride():
                    raise ValueError("Persistent KV binding changed")
            else:
                dst.copy_(src)
        if self.reference_mode:
            return reference(self.variant, self.buffers).to(inputs[0].dtype)
        output = self.call()
        if not self.checked:
            expected = reference(self.variant, self.buffers)
            atol = 8e-3 if self.variant.startswith("sparse") else 3e-3
            torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=atol)
            self.error = ((output.float() - expected).norm()
                          / expected.norm().clamp_min(1e-30)).item()
            if self.error >= 1e-2:
                raise AssertionError(f"Attention relative L2 error too large: {self.error}")
            self.checked = True
        return output


def grouped_attention(core, query, key, value, heads):
    if query.shape[1] == heads:
        return core(query, key, value)
    if query.shape[1] % heads:
        raise ValueError("Model heads must be divisible by exported kernel heads")
    output = torch.empty_like(query)
    for start in range(0, query.shape[1], heads):
        group = slice(start, start + heads)
        # The exported core reuses its output buffer, so consume each result immediately.
        output[:, group].copy_(core(query[:, group], key[:, group], value[:, group]))
    return output


@torch.inference_mode()
def build_decode_model(core_factory, device="cuda", variant="decode", kv_length=None,
                       implementation="nautilus", query_length=16):
    from transformers import LlamaConfig, LlamaModel
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    # Match the existing prefill benchmark's Llama-7B backbone, without unused vocabulary weights.
    shapes = [tuple(p["shape"]) for p in metadata(variant, kv_length, query_length)["params"][:3]]
    _, export_heads, query_length, _ = shapes[0]
    length = shapes[1][2]
    config = LlamaConfig(vocab_size=16, attn_implementation="sdpa",
                         max_position_embeddings=max(4096, length))
    with torch.device(device):
        model = LlamaModel(config).eval().to(dtype=torch.float16)
    model.paper_weights_sha256 = tensor_hash(model.parameters())
    prefix = torch.randn((1, length - query_length, config.hidden_size),
                         device=device, dtype=torch.float16)
    cache = model(inputs_embeds=prefix, use_cache=True).past_key_values
    positions = None
    if variant in ("decode", "mtpc"):
        from argus.paper_baselines.e2e_native_cache import build_native_cache

        cache = build_native_cache(
            config, cache, length, export_heads,
            "bshd" if implementation in ("tileops", "tileops_bound", "tilelang", "baseline")
            else "bhsd",
        )
        positions = torch.arange(length - query_length, length, device=device)
    model.paper_cache_implementation = "persistent_native" if positions is not None else "dynamic"
    model.paper_cache_layout = getattr(cache, "layout", "bhsd")
    token = torch.randn((1, query_length, config.hidden_size), device=device, dtype=torch.float16)
    model.paper_inputs_sha256 = tensor_hash((prefix, token))

    def attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **_):
        if dropout or scaling != 128 ** -0.5:
            raise ValueError("Paper decode requires dropout=0 and scale=128**-0.5")
        if variant in ("decode", "mtpc"):
            # HF forwards this private grouped cache directly to the registered attention hook.
            layer = cache.layers[module.layer_idx]
            groups = layer.groups_bhsd()
            if key is not layer.keys or value is not layer.values:
                raise ValueError("Native cache update did not return its persistent tensors")
            if not hasattr(module, "paper_cores"):
                module.paper_cores = [
                    core_factory(variant, (query[:, g * export_heads:(g + 1) * export_heads], k, v),
                                 persistent_kv=True)
                    for g, (k, v) in enumerate(groups)
                ]
            if len(groups) == 1:
                key, value = groups[0]
                return module.paper_cores[0](query, key, value).transpose(1, 2), None
            output = torch.empty_like(query)
            for g, (core, (k, v)) in enumerate(zip(module.paper_cores, groups)):
                group = slice(g * export_heads, (g + 1) * export_heads)
                output[:, group].copy_(core(query[:, group], k, v))
            return output.transpose(1, 2), None
        raise ValueError(f"Unsupported cached model variant: {variant}")

    ALL_ATTENTION_FUNCTIONS.register("paper_e2e", attention)
    model.config._attn_implementation = "paper_e2e"

    def inputs():
        if positions is not None:
            return {"inputs_embeds": token, "past_key_values": cache, "use_cache": True,
                    "cache_position": positions, "position_ids": positions.unsqueeze(0)}
        raise ValueError(f"Missing explicit cache positions for {variant}")

    return (model, inputs, f"Llama-7B backbone, 32 layers, MHA, Q={query_length}, "
            f"prefix={length - query_length}, total KV={length}, variant={variant}")


def build_model(variant, factory, kv_length=None, implementation="nautilus", query_length=16):
    if variant in ("decode", "mtpc"):
        return build_decode_model(factory, variant=variant, kv_length=kv_length,
                                  implementation=implementation, query_length=query_length)
    from argus.paper_baselines.e2e_sparse import build_sparse_model

    model, inputs, description = build_sparse_model(variant, factory)
    return model, lambda: inputs, description


def measure(args):
    query_length = getattr(args, "query_length", 16)
    use_graph = getattr(args, "cudagraph", False)
    graph_backends = {"decode": ("tileops_bound", "nautilus", "flex"),
                      "mtpc": ("tileops", "nautilus", "flex")}
    if use_graph and args.implementation not in graph_backends.get(args.variant, ()):
        raise ValueError("Unsupported variant/backend for whole-step graph replay")
    if not torch.cuda.is_available():
        raise RuntimeError("Run inside an allocated GPU job")
    if args.implementation == "nautilus" and (
        platform.machine() != "aarch64" or torch.cuda.get_device_capability() != (9, 0)
    ):
        raise RuntimeError("Exported paper kernels require aarch64 + SM90")
    torch.manual_seed(0)
    # HF attention guards specialize on each layer's cache; the default limit is only eight.
    torch._dynamo.config.recompile_limit = 300
    cores = []

    def factory(variant, inputs, persistent_kv=False):
        core = AttentionCore(variant, inputs, args.implementation, args.kv_length, persistent_kv,
                             query_length=query_length)
        cores.append(core)
        return core

    print(f"Building {args.variant}/{args.implementation} model and prefix cache", flush=True)
    model, make_inputs, description = build_model(args.variant, factory, args.kv_length,
                                                 args.implementation, query_length=query_length)
    with torch.inference_mode():
        print("Checking reference and backend model outputs", flush=True)
        expected = model(**make_inputs()).last_hidden_state.float().clone()
        if not cores:
            raise RuntimeError("Model never invoked the paper attention adapter")
        for core in cores:
            core.reference_mode = False
        actual = model(**make_inputs()).last_hidden_state.float()
        relative_l2 = ((actual - expected).norm() / expected.norm().clamp_min(1e-30)).item()
        print(f"Model relative L2={relative_l2}, "
              f"max abs error={(actual - expected).abs().max().item()}", flush=True)
        if not math.isfinite(relative_l2) or relative_l2 >= 1e-2:
            raise AssertionError(f"Model relative L2 error too large: {relative_l2}")

        # TVM calls stay outside Dynamo; projections, MLPs and norms use the existing compile path.
        print("Compiling and warming up the model", flush=True)
        compiled = torch.compile(model, dynamic=False, fullgraph=False)
        for _ in range(args.warmup):
            result = compiled(**make_inputs()).last_hidden_state
        torch.cuda.synchronize()
        compiled_relative_l2 = ((result.float() - expected).norm()
                                / expected.norm().clamp_min(1e-30)).item()
        if not math.isfinite(compiled_relative_l2) or compiled_relative_l2 >= 1e-2:
            raise AssertionError(f"Compiled model relative L2 too large: {compiled_relative_l2}")

        replay, graph_validation = None, None
        if use_graph:
            from argus.paper_baselines.e2e_graph import capture_decode_graph

            print("Capturing and checking whole-step CUDA Graph", flush=True)
            replay, graph_validation = capture_decode_graph(
                compiled, make_inputs(), cores, expected, args.implementation, args.variant,
            )
            for _ in range(args.warmup):
                replay()
            torch.cuda.synchronize()

        from argus.utils import profiler_ctx

        samples = []
        print(f"Measuring {args.repeat} synchronized forwards", flush=True)
        with profiler_ctx() if args.profile_child else nullcontext():
            for i in range(args.repeat):
                inputs = make_inputs()  # Cache reconstruction and its copies are outside timing.
                torch.cuda.synchronize()
                with torch.cuda.nvtx.range(f"paper_e2e_{args.variant}_{args.implementation}[{i}]"):
                    start = perf_counter()
                    result = replay() if use_graph else compiled(**inputs).last_hidden_state
                    torch.cuda.synchronize()
                    samples.append((perf_counter() - start) * 1000)
                del inputs  # Release this cache before constructing the next long-context copy.

        if use_graph:
            from argus.paper_baselines.e2e_graph import audit_graph_replay

            print("Auditing captured kernels outside formal timing", flush=True)
            graph_validation.update(audit_graph_replay(
                replay, args.kv_length or (65536 if args.variant == "mtpc" else 512),
                args.output.with_suffix(".graph.trace.json"), args.implementation, args.variant,
            ))

    import transformers

    payload = {
        "variant": args.variant, "implementation": args.implementation,
        "query_length": cores[0].buffers[0].shape[2] if args.variant in ("decode", "mtpc") else None,
        "model": description, "config": model.config.to_dict(), "seed": 0,
        "scope": "compiled backbone forward, model projections, attention staging, MLP, norm; "
                 "cache reset/setup, tokenizer, embedding lookup, LM head and sampling excluded",
        "metric": "synchronized wall latency in ms; includes CPU dispatch and GPU completion",
        "l2": "steady state; no explicit flush", "warmup": args.warmup,
        "samples_ms": samples, "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples), "min_ms": min(samples),
        "correct": True, "relative_l2": relative_l2,
        "compiled_relative_l2": compiled_relative_l2,
        "max_abs_error": (actual - expected).abs().max().item(),
        "attention_relative_l2": [core.error for core in cores],
        "baseline_source": cores[0].source,
        "cache_implementation": getattr(model, "paper_cache_implementation", None),
        "cache_layout": getattr(model, "paper_cache_layout", None),
        "persistent_kv_bindings": sum(core.persistent_kv for core in cores),
        "full_kv_staging_bytes_per_forward": 0 if args.variant in ("decode", "mtpc") else None,
        "weights_sha256": getattr(model, "paper_weights_sha256", None),
        "inputs_sha256": getattr(model, "paper_inputs_sha256", None),
        "gpu": torch.cuda.get_device_name(), "host": platform.node(),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "repeat": args.repeat,
        "export_directory": str(export_path(args.variant, args.kv_length, query_length)),
        "kernel_sha256": hashlib.sha256(
            (export_path(args.variant, args.kv_length, query_length) / "kernel.so").read_bytes()
        ).hexdigest(),
        "packages": {name: {"version": getattr(sys.modules[name], "__version__", None),
                            "path": getattr(sys.modules[name], "__file__", None)}
                     for name in ("tvm", "tilelang", "triton") if name in sys.modules},
        "measurement_complete": True,
    }
    if use_graph:
        payload.update(execution="cuda_graph_replay", graph_validation=graph_validation,
                       metric="synchronized whole-step CUDA Graph replay wall latency in ms",
                       scope="whole backbone graph replay including new KV writes, projections, "
                             "attention, MLP and norm; prefix/input setup, capture, LM head and "
                             "sampling excluded")
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument(
        "--implementation", choices=("nautilus", "tileops", "tileops_bound", "tilelang",
                                      "flex", "sdpa", "baseline"),
        default="nautilus",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kv-length", type=int, default=None,
                        help="Select an exported dense length; default decode=512, mtpc=65536")
    parser.add_argument("--query-length", type=int, choices=(2, 4, 8, 16), default=16,
                        help="MTP query tokens; decode remains Q=1")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--cudagraph", action="store_true",
                        help="Whole-step replay for fixed-shape decode/MTP")
    parser.add_argument("--profiler", choices=("nsys",))
    parser.add_argument("--profile-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.query_length != 16 and args.variant != "mtpc":
        parser.error("--query-length only applies to mtpc")
    if args.repeat < 1 or args.warmup < 1:
        parser.error("--repeat and --warmup must be positive")
    if args.kv_length is not None and args.kv_length <= 0:
        parser.error("--kv-length must be positive")
    supported = {"decode": ("nautilus", "tileops", "tileops_bound", "tilelang", "flex",
                            "sdpa", "baseline"),
                 "mtpc": ("nautilus", "tileops", "flex"),
                 "sparse": ("nautilus", "tilelang", "baseline"),
                 "sparse_split": ("nautilus", "tilelang", "baseline")}
    if args.implementation not in supported[args.variant]:
        parser.error(f"Unsupported backend {args.implementation} for {args.variant}")
    graph_backends = {"decode": ("tileops_bound", "nautilus", "flex"),
                      "mtpc": ("tileops", "nautilus", "flex")}
    if args.cudagraph and args.implementation not in graph_backends.get(args.variant, ()):
        parser.error("Unsupported variant/backend for --cudagraph")
    if args.kv_length is not None and args.variant.startswith("sparse"):
        parser.error("--kv-length only applies to decode and mtpc")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if (args.output.with_suffix(".json").exists()
            or (not args.profile_child and args.output.with_suffix(".nsys-rep").exists())):
        parser.error("Output already exists; choose a new --output prefix")
    if args.profiler and not args.profile_child:
        from argus.utils import relaunch_with_profiler

        relaunch_with_profiler(
            args.output, "nsys", False,
            ["-m", "scripts.bench_paper_e2e", *sys.argv[1:], "--profile-child"],
            wait_for_cuda_profiler=True,
        )
    else:
        measure(args)


if __name__ == "__main__":
    main()
