import logging
from pathlib import Path
from typing import Sequence

import torch

from . import BenchmarkOperator, BenchmarkRunner, triton
from . import operators as ops
from .flashattn import TriDaoRunner
from .flashinfer import FlashInferRunner
from .pytorch.flex import FlexAlibiRunner, FlexAttentionRunner
from .pytorch.sdpa import AttentionAlgo, TorchDecodeGQARunner, TorchSDPARunner
from .utils import profiler_ctx

logger = logging.getLogger(__name__)


def torch_softcap(softcap: float = 10.0):
    return lambda score, b, h, q_idx, kv_idx: torch.tanh(score / softcap) * softcap


def flex_windowed_mask_cond(left_window_size: int):
    return lambda b, h, q_idx, kv_idx: (q_idx >= kv_idx) & (q_idx - kv_idx < left_window_size)


RUNNERS: dict[BenchmarkOperator, Sequence[BenchmarkRunner]] = {
    ops.PF_GLOBAL: [
        FlexAttentionRunner(ops.PF_GLOBAL),
        TorchSDPARunner(ops.PF_GLOBAL, AttentionAlgo.FLASH, False),
        TorchSDPARunner(ops.PF_GLOBAL, AttentionAlgo.CUDNN, False),
        triton.OpenAITritonRunner(ops.PF_GLOBAL, False),
        triton.TriDaoTritonRunner(ops.PF_GLOBAL, False),
        TriDaoRunner(ops.PF_GLOBAL, causal=False),
        FlashInferRunner(ops.PF_GLOBAL, is_causal=False, is_decode=False),
    ],
    ops.PF_CAUSAL: [
        FlexAttentionRunner(ops.PF_CAUSAL, mask_expr=ops.causal_mask_cond),
        TorchSDPARunner(ops.PF_CAUSAL, AttentionAlgo.FLASH, True),
        TorchSDPARunner(ops.PF_CAUSAL, AttentionAlgo.CUDNN, True),
        triton.OpenAITritonRunner(ops.PF_CAUSAL, True),
        triton.TriDaoTritonRunner(ops.PF_CAUSAL, True),
        TriDaoRunner(ops.PF_CAUSAL, causal=True),
        FlashInferRunner(ops.PF_CAUSAL, is_causal=True, is_decode=False),
    ],
    ops.PF_GQA: [
        FlexAttentionRunner(ops.PF_GQA, mask_expr=ops.causal_mask_cond, enable_gqa=True),
        TorchSDPARunner(ops.PF_GQA, AttentionAlgo.FLASH, True, is_gqa=True),
        TorchSDPARunner(ops.PF_GQA, AttentionAlgo.CUDNN, True, is_gqa=True),
        triton.OpenAITritonRunner(ops.PF_GQA, True),
        triton.TriDaoTritonRunner(ops.PF_GQA, True),
        # The following 2 runners don't make a distinction between GQA and non-GQA.
        TriDaoRunner(ops.PF_GQA, causal=True),
        FlashInferRunner(ops.PF_GQA, is_causal=True, is_decode=False),
        # TVM runs GQA by simply repeating the key and value tensors.
        # TVMPrefillRunner(ops.PF_GQA, True),
    ],
    ops.PF_ALIBI: [
        FlexAlibiRunner(ops.PF_ALIBI, is_decode=False),
        # TriDao Triton could support Alibi, but runs out of SMEM.
        TriDaoRunner(ops.PF_ALIBI, has_alibi=True),
    ],
    # This "softcap" operator is causal + GQA + softcap.
    ops.PF_SOFTCAP: [
        FlexAttentionRunner(
            ops.PF_SOFTCAP,
            score_mod=torch_softcap(softcap=ops.PF_SOFTCAP.softcap),
            mask_expr=ops.causal_mask_cond,
            enable_gqa=True,
        ),
    ],
    # This "windowed" operator is window + GQA + softcap.
    ops.PF_WINDOWED: [
        FlexAttentionRunner(
            ops.PF_WINDOWED,
            score_mod=torch_softcap(softcap=ops.PF_SOFTCAP.softcap),
            mask_expr=flex_windowed_mask_cond(ops.PF_WINDOWED.window_size),
            enable_gqa=True,
        ),
        TriDaoRunner(
            ops.PF_WINDOWED,
            window_size=(ops.PF_WINDOWED.window_size - 1, 0),
            softcap=ops.PF_SOFTCAP.softcap,
        ),
    ],
    # Decoding attention begins.
    ops.DC_CAUSAL: [
        FlexAttentionRunner(ops.DC_CAUSAL),
        # NOTE 1: In PyTorch, neither the "FlashAttn" nor "Cudnn" backends support decoding. Decoding is
        # handled by the so-called "mem efficient" backend.
        # NOTE 2: the `is_causal` flag of PyTorch SDPA should be set to False for decoding. If it's True,
        # only 1 token from K/V is attended to (which does not produce a meaningful output).
        TorchSDPARunner(ops.DC_CAUSAL, AttentionAlgo.MEM_EFFICIENT, False, False),
        TriDaoRunner(ops.DC_CAUSAL, causal=True),
        FlashInferRunner(ops.DC_CAUSAL, is_causal=False, is_decode=True),
        # TVMDecodeRunner(ops.DC_CAUSAL),
        triton.XformerDecodeRunner(ops.DC_CAUSAL),
    ],
    ops.DC_GQA: [
        FlexAttentionRunner(ops.DC_GQA, enable_gqa=True),
        TriDaoRunner(ops.DC_GQA, causal=True),
        FlashInferRunner(ops.DC_GQA, is_causal=False, is_decode=True),
        TorchDecodeGQARunner(ops.DC_GQA),
        # TVMDecodeRunner(ops.DC_GQA),
    ],
    ops.DC_ALIBI: [
        FlexAlibiRunner(ops.DC_ALIBI, is_decode=True),
        TriDaoRunner(ops.DC_ALIBI, has_alibi=True),
    ],
    ops.DC_SOFTCAP: [
        FlexAttentionRunner(
            ops.DC_SOFTCAP,
            score_mod=torch_softcap(softcap=ops.DC_SOFTCAP.softcap),
            enable_gqa=True,
        ),
    ],
}
RUNNERS_BY_OP_NAME = {op.name: (op, runners) for op, runners in RUNNERS.items()}


def run_operator(operator: str, shape: list[int], n_repeats: int, low_mem_overhead: bool):
    op, runners = RUNNERS_BY_OP_NAME[operator]
    assert len(runners) >= 1, f"No runners for operator {operator}"
    # The complete shape (often 6D for attention) is a "shape signature". It isn't the shape of any actual input.
    shape_sig = op.complete_shape(tuple(shape))
    inputs = op.create_inputs(shape_sig)
    # NOTE: between a kernel run and reading the result, do not run any other kernels.
    # CUDAGraph may complain about the result being overwritten.
    # NOTE: we normally iterate over all runners (set-up, run once, compare results) first,
    # then run the actual repeats. It allows the profiler to capture only the repeats continuously
    # in a single report. This has a memory overhead of len(runners) * mem_per_runner.
    # When `low_mem_overhead` is True, we do everything with a single runner and free its memory
    # before moving on to the next runner. The profiles generated this way will be longer.
    ref_output = None
    runners_: list[BenchmarkRunner] = []
    for runner in runners:
        logger.info("starting %s", runner)
        if (mem_reserved_mb := torch.cuda.memory_reserved() / 1024**2) > 1024:
            logger.info("memory usage: %.3fMB", mem_reserved_mb)
        # Set up the runner. If it's not supported or fails to run, skip it and free its memory.
        if not runner.setup(*inputs):
            logger.warning(f"{operator}: {runner} does not support this setup")
            runner.free_members_memory()
            continue
        try:
            runner.run()
        except Exception as e:
            logger.warning(f"{operator}: {runner} failed with exception: {e}")
            runner.free_members_memory()
            continue
        runners_.append(runner)
        this_output = runner.get_output()
        if ref_output is None:
            ref_output = this_output
        else:
            shape_match = ref_output.shape == this_output.shape
            assert shape_match, f"{operator}: {runner} shape mismatches against {runners[0]}"
            if not torch.allclose(ref_output.cpu(), this_output.cpu(), atol=1e-2, rtol=0):
                logger.warning(f"{operator}: {runner} result mismatch against {runners[0]}")
        del this_output
        if low_mem_overhead:
            # Run the repeats here to reduce memory overhead.
            runner.repeat(n_repeats)
            runner.free_members_memory()
    if not low_mem_overhead:
        # If not in low-mem-overhead mode, run the repeats all at once with profiler on.
        with profiler_ctx():
            for runner in runners_:
                runner.repeat(n_repeats)


# No support for running multiple operators at once -- the nvtx markers won't be unique.
def profile_operator(
    operator: str,
    shape: list[int],
    profiler: str | None,
    repeat: int,
    low_mem_overhead: bool,
    gpu_metrics: bool,
):
    from .utils import relaunch_with_profiler

    if profiler is None:
        run_operator(operator, shape, repeat, low_mem_overhead)
        return
    if profiler == "ncu" and repeat > 1:
        raise ValueError("ncu profiler does not support repeat > 1")
    # Get the operator, and use it to get the shape signature.
    op, _ = RUNNERS_BY_OP_NAME[operator]
    shape_str = ",".join(str(x) for x in shape)
    shape_sig = op.complete_shape(tuple(shape))
    output_prefix = Path("logs/profiles/")
    output_prefix.mkdir(parents=True, exist_ok=True)
    output_prefix = output_prefix / f"{operator}-{','.join(str(x) for x in shape_sig)}"
    relaunch_args = ["-m", __spec__.name, "profile", operator, shape_str, "--repeat", str(repeat)]
    relaunch_with_profiler(output_prefix, profiler, gpu_metrics, relaunch_args)


def parse_args():
    import argparse

    def parse_shape(s: str) -> list[int]:
        return list(int(x) for x in s.split(","))

    parser = argparse.ArgumentParser()
    operators = {op.name for op in RUNNERS.keys()}
    parser.add_argument("operator", type=str, choices=operators)
    parser.add_argument("shape", type=parse_shape)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--profiler", type=str, choices=["ncu", "nsys", "rocprof"])
    parser.add_argument("--low-mem-overhead", action="store_true")
    parser.add_argument("--gpu-metrics", action="store_true")
    parser.set_defaults(
        func=lambda args: profile_operator(
            args.operator,
            args.shape,
            args.profiler,
            args.repeat,
            args.low_mem_overhead,
            args.gpu_metrics,
        )
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    args.func(args)
