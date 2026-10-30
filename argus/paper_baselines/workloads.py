"""Inputs and independent FP32 references for exported paper workloads."""

import hashlib
import json
from pathlib import Path

import torch


PAPER_ROOT = Path(__file__).resolve().parents[2] / "paper_kernels"
LEGACY_PAPER_ROOT = PAPER_ROOT.with_name("paper_kernels.previous_20261026")
VARIANTS = ("prefill", "decode", "mtp", "sparse", "sparse_split")
E2E_VARIANTS = VARIANTS + ("mtpc",)


class NautilusUnavailable(RuntimeError):
    """Neither the exported module nor its original scheduled source can run here."""


def resolve_export(case, root=None):
    """Use the same updated export precedence for kernel and model benchmarks."""
    root = PAPER_ROOT if root is None else root
    for backend in ("tawa", "tilelang", "tawa_mha"):
        base = root / case / backend
        for path in (base / "tilelang_auto/best/export", base / "best/export"):
            if (path / "kernel.so").is_file():
                return path
    raise FileNotFoundError(f"No exported kernel for {case}")


def export_path(variant, kv_length=None, query_length=16):
    if variant not in E2E_VARIANTS:
        raise ValueError(f"Unknown paper variant: {variant}")
    backend = "tawa" if variant == "prefill" else "tilelang"
    if variant == "decode":
        return resolve_export(f"decode_L{kv_length or 512}")
    elif variant in ("mtp", "mtpc"):
        if query_length not in (2, 4, 8, 16, 32, 64):
            raise ValueError(f"Unsupported MTP query length: {query_length}")
        path = PAPER_ROOT / f"{variant}_Q{query_length}_L{kv_length or 65536}/tilelang/best/export"
    else:
        path = PAPER_ROOT / variant / backend / "best/export"
    if not path.exists() and kv_length is None and query_length == 16:
        path = LEGACY_PAPER_ROOT / variant / backend / "best/export"
    return path


def metadata(variant, kv_length=None, query_length=16):
    path = export_path(variant, kv_length, query_length)
    meta = path / "meta.json"
    if meta.exists():
        return json.loads(meta.read_text())
    # Sequence-sweep exports predate meta.json; their schedule has the exact ABI.
    import re

    text = (path / "schedule.py").read_text()
    params = []
    match = re.search(r"def neptune_attention\((.*?)\):", text, re.S)
    if match is None:
        raise FileNotFoundError(f"No meta.json or ABI in {path}")
    for name, shape, dtype in re.findall(
        r"(\w+): T\.Buffer\(\(([^)]*)\), \"(\w+)\"\)", match.group(1)
    ):
        params.append({"name": name, "shape": [int(x.strip()) for x in shape.split(",")],
                       "dtype": dtype})
    if len(params) != 4:
        raise ValueError(f"Expected four dense kernel parameters in {path}")
    return {"variant": variant, "backend": "tilelang", "params": params}


def structured_indices(tokens=2048, kv_rows=512, topk=2048):
    """Vectorized round-robin generator from upstream scripts/test_sparse_attn.py."""
    if topk % 4 or min(tokens, kv_rows, topk) <= 0:
        raise ValueError("Positive dimensions and topk divisible by four are required")
    t = torch.arange(tokens)[:, None]
    cur = t.clamp_max(kv_rows - 1)
    j = torch.arange(topk // 4)[None, :]
    local = torch.where((j < 256) & (j <= cur), cur - j, -1)
    stride = torch.where((j < 256) & (8 * j <= cur), cur - 8 * j, -1)
    step = max(kv_rows // 256, 1)
    anchors = torch.where((j < 256) & (j * step < kv_rows), j * step, -1)
    retrieval = torch.where(j < ((t * 17) % 128), (t * 13 + j) % kv_rows, -1)
    return torch.stack((local, stride, anchors.expand_as(local), retrieval), dim=-1).reshape(
        tokens, topk
    ).to(torch.int32)


def _cpu_inputs(variant):
    torch.manual_seed(0)
    if variant in ("sparse", "sparse_split"):
        qn = torch.randn((2048, 16, 512), dtype=torch.bfloat16)
        qp = torch.randn((2048, 16, 64), dtype=torch.bfloat16)
        ckv = torch.randn((8, 64, 512), dtype=torch.bfloat16)
        kpe = torch.randn((8, 64, 64), dtype=torch.bfloat16)
        indices = structured_indices()
        if variant == "sparse":
            inputs = [torch.cat((qn, qp), dim=-1), torch.cat(
                (ckv.reshape(512, 512), kpe.reshape(512, 64)), dim=-1
            ), indices]
        else:
            inputs = [qn, qp, ckv, kpe, indices]
    else:
        inputs = [torch.randn(p["shape"], dtype=getattr(torch, p["dtype"]))
                  for p in metadata(variant)["params"][:-1]]
    return inputs


def make_inputs(variant, device="cuda", input_file=None):
    """Share actual CPU input bytes: Torch versions can change seeded RNG output."""
    if input_file is not None and input_file.exists():
        inputs = torch.load(input_file, map_location="cpu", weights_only=True)
    else:
        inputs = _cpu_inputs(variant)
        if input_file is not None:
            temporary = input_file.with_suffix(".tmp")
            torch.save(inputs, temporary)
            temporary.replace(input_file)
    params = metadata(variant)["params"][:-1]
    if not isinstance(inputs, list) or len(inputs) != len(params):
        raise ValueError("Stored input count does not match the export")
    for x, p in zip(inputs, params):
        if (not isinstance(x, torch.Tensor) or list(x.shape) != p["shape"]
                or x.dtype != getattr(torch, p["dtype"])):
            raise ValueError("Stored input shape/dtype does not match the export")
    digest = hashlib.sha256()
    for x in inputs:
        digest.update(x.contiguous().view(torch.uint8).numpy().tobytes())
    return [x.to(device).contiguous() for x in inputs], digest.hexdigest()


def validate_inputs(variant, inputs, kv_length=None, query_length=16):
    params = metadata(variant, kv_length, query_length)["params"][:-1]
    if len(inputs) != len(params):
        raise ValueError(f"{variant} requires {len(params)} inputs")
    for x, p in zip(inputs, params):
        if tuple(x.shape) != tuple(p["shape"]) or x.dtype != getattr(torch, p["dtype"]):
            raise ValueError(f"{variant}/{p['name']} requires {p['shape']} {p['dtype']}")
        if x.device.type != "cuda" or not x.is_contiguous():
            raise ValueError("Exported paper kernels require contiguous CUDA inputs")
    if len({x.device for x in inputs}) != 1:
        raise ValueError("All inputs must share a CUDA device")


@torch.no_grad()
def sparse_reference(q, kv, indices, value_dim=512, chunk=32, dtype=torch.float32):
    """Gather preserves repeated indices; -1 is ignored, with no causal masking."""
    result = torch.empty((*q.shape[:-1], value_dim), device=q.device, dtype=dtype)
    if indices.min() < -1 or indices.max() >= kv.shape[0]:
        raise ValueError("Sparse indices must be -1 or a valid KV row")
    for begin in range(0, q.shape[0], chunk):
        idx = indices[begin:begin + chunk].long()
        gathered = kv[idx.clamp_min(0)].to(dtype)
        logits = q[begin:begin + chunk].to(dtype) @ gathered.transpose(1, 2)
        logits *= value_dim ** -0.5
        logits.masked_fill_((idx < 0)[:, None, :], float("-inf"))
        weights = torch.softmax(logits, dim=-1).nan_to_num(0)
        result[begin:begin + chunk] = weights @ gathered[:, :, :value_dim]
    return result


def reference(variant, inputs):
    torch.backends.cuda.matmul.allow_tf32 = False
    if variant in ("sparse", "sparse_split"):
        if variant == "sparse":
            q, kv, indices = inputs
        else:
            qn, qp, ckv, kpe, indices = inputs
            q = torch.cat((qn, qp), dim=-1)
            kv = torch.cat((ckv.reshape(512, 512), kpe.reshape(512, 64)), dim=-1)
        return sparse_reference(q, kv, indices)
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        if variant == "mtpc":
            q, k, v = (x.float() for x in inputs)
            q_len, kv_len = q.shape[-2], k.shape[-2]
            allowed = torch.arange(kv_len, device=q.device)[None, :] <= (
                kv_len - q_len + torch.arange(q_len, device=q.device)[:, None]
            )
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed[None, None]
            )
        return torch.nn.functional.scaled_dot_product_attention(*(x.float() for x in inputs))


def build_exported(variant, inputs, kv_length=None, query_length=16):
    validate_inputs(variant, inputs, kv_length, query_length)
    try:
        import tvm
    except ModuleNotFoundError as exc:
        if exc.name != "tvm":
            raise
        import tilelang

        tvm = tilelang.tvm

    if variant == "prefill":
        # The legacy callback checks this attribute without importing its module.
        import triton.tools.tensor_descriptor  # noqa: F401
    out = metadata(variant, kv_length, query_length)["params"][-1]
    output = torch.empty(out["shape"], dtype=getattr(torch, out["dtype"]), device=inputs[0].device)
    try:
        module = tvm.runtime.load_module(str(export_path(variant, kv_length, query_length) / "kernel.so"))
        entry = module["main"]
    except Exception as load_error:
        if hasattr(tvm, "nd"):
            raise
        try:
            import tilelang
            from tvm.script import ir, tirx

            scheduled = tvm.script.from_source(
                (export_path(variant, kv_length, query_length) / "schedule.py").read_text(),
                extra_vars={"I": ir, "T": tirx},
            )
            kernel = tilelang.compile(next(iter(scheduled.functions.values())), out_idx=[])
        except Exception as compile_error:
            raise NautilusUnavailable(
                f"Exported module: {load_error}\nScheduled source: {compile_error}"
            ) from compile_error

        def recompiled():
            kernel(*inputs, output)
            return output

        return recompiled
    from_dlpack = tvm.nd.from_dlpack if hasattr(tvm, "nd") else tvm.runtime.from_dlpack
    args = [from_dlpack(x) for x in (*inputs, output)]

    def run(_module=module):
        # The host entry must keep its imported CUDA modules alive.
        entry(*args)
        return output

    return run
