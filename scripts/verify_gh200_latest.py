"""Check the figure baselines on GH200 against FP32 attention, in separate processes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

BACKENDS = (
    "flex", "flex_fp8", "tawa", "tawa_fp8", "openai", "tri_dao",
    "tilelang", "helion", "flashattn", "sdpa", "cudnn", "flashinfer",
)
OPERATORS = ("global", "causal", "gqa")


def check(backend, operator):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn import functional as F

    from argus import operators as ops

    assert torch.cuda.is_available(), "Run on an allocated GH200 node"
    assert "GH200" in torch.cuda.get_device_name(), torch.cuda.get_device_name()
    torch.manual_seed(0)
    op = {"global": ops.PF_GLOBAL, "causal": ops.PF_CAUSAL, "gqa": ops.PF_GQA}[operator]
    causal = operator != "global"
    gqa = operator == "gqa"
    q = torch.randn((1, 4, 256, 128), dtype=torch.float16, device="cuda")
    k = torch.randn((1, 1 if gqa else 4, 256, 128), dtype=q.dtype, device=q.device)
    v = torch.randn_like(k)
    if backend.endswith("_fp8"):
        q, k, v = (x.to(torch.float8_e4m3fn).to(torch.float16) for x in (q, k, v))
    mask = ops.causal_mask_cond_4d if causal else None
    if backend.startswith("flex"):
        from argus.pytorch.flex import FlexAttentionFP8Runner, FlexAttentionRunner
        cls = FlexAttentionFP8Runner if backend.endswith("_fp8") else FlexAttentionRunner
        runner = cls(op, mask_expr=mask, enable_gqa=gqa)
    elif backend.startswith("tawa"):
        from argus.tawa import TawaPrefillFP8Runner, TawaPrefillRunner
        cls = TawaPrefillFP8Runner if backend.endswith("_fp8") else TawaPrefillRunner
        runner = cls(op, is_causal=causal)
    elif backend in ("openai", "tri_dao"):
        from argus.triton_runners import OpenAITritonRunner, TriDaoTritonRunner
        cls = OpenAITritonRunner if backend == "openai" else TriDaoTritonRunner
        runner = cls(op, is_causal=causal)
    elif backend == "tilelang":
        from argus.tilelang_pypi import TileLangPrefillRunner
        runner = TileLangPrefillRunner(op, is_causal=causal, enable_gqa=gqa)
    elif backend == "helion":
        from argus.helion import HelionRunner
        runner = HelionRunner(op)
    elif backend == "flashattn":
        from argus.flashattn import TriDaoRunner
        runner = TriDaoRunner(op, causal=causal)
    elif backend == "flashinfer":
        from argus.flashinfer import FlashInferRunner
        runner = FlashInferRunner(op, is_causal=causal, is_decode=False)
    else:
        from argus.pytorch.sdpa import AttentionAlgo, TorchSDPARunner
        algo = AttentionAlgo.CUDNN if backend == "cudnn" else AttentionAlgo.FLASH
        runner = TorchSDPARunner(op, algo, causal, is_gqa=gqa)
    assert runner.setup(q, k, v), f"{backend} setup failed"
    runner.run()
    torch.cuda.synchronize()
    out = runner.get_output().to(device="cuda", dtype=torch.float32)
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), is_causal=causal, enable_gqa=gqa,
        )
    assert out.shape == ref.shape and torch.isfinite(out).all()
    relative_l2 = ((out - ref).norm() / ref.norm()).item()
    if backend.endswith("_fp8"):
        assert relative_l2 < 0.05, relative_l2
    else:
        torch.testing.assert_close(out, ref, atol=3e-3, rtol=1e-2)
    return {"max_abs_error": (out - ref).abs().max().item(), "relative_l2": relative_l2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("all", *BACKENDS), default="all")
    parser.add_argument("--operator", choices=OPERATORS)
    args = parser.parse_args()
    root = Path(os.environ["ARGUS_GH200_WORK"]) / "validation"
    root.mkdir(parents=True, exist_ok=True)
    if args.operator:
        assert args.backend != "all"
        result = {"backend": args.backend, "operator": args.operator}
        try:
            result.update(check(args.backend, args.operator), passed=True)
        except Exception:
            result.update(passed=False, error=traceback.format_exc())
        (root / f"{args.backend}-{args.operator}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result), flush=True)
        return 0 if result["passed"] else 1
    backends = BACKENDS if args.backend == "all" else (args.backend,)
    failed = []
    for backend in backends:
        for operator in OPERATORS:
            log_path = root / f"{backend}-{operator}.log"
            result_path = root / f"{backend}-{operator}.json"
            result_path.unlink(missing_ok=True)
            print(f"Checking {backend}/{operator}: {log_path}", flush=True)
            with log_path.open("w") as log:
                try:
                    result = subprocess.run(
                        [sys.executable, __file__, "--backend", backend,
                         "--operator", operator], stdout=log, stderr=subprocess.STDOUT,
                        timeout=600,
                    )
                    passed = result.returncode == 0
                except subprocess.TimeoutExpired:
                    passed = False
            if not passed and not result_path.exists():
                result_path.write_text(json.dumps({
                    "backend": backend, "operator": operator, "passed": False,
                    "error": "verification child exited without a result record",
                }, indent=2) + "\n")
            if not passed:
                failed.append(f"{backend}/{operator}")
    print(json.dumps({"failed": failed}), flush=True)
    return bool(failed)


if __name__ == "__main__":
    sys.exit(main())
