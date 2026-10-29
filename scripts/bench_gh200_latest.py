"""GH200 profile entrypoint with every current non-TVM figure baseline enabled."""

from __future__ import annotations

import json
import sys

from argus import bench_new as bench
from argus import operators as ops
from argus.flashattn import TriDaoRunner
from argus.flashinfer import FlashInferRunner
from argus.helion import HelionRunner
from argus.neptune_tilelang_pypi import NeptuneTileLangPrefillRunnerBase
from argus.pytorch.flex import FlexAttentionRunner
from argus.pytorch.sdpa import AttentionAlgo, TorchSDPARunner
from argus.tawa import TawaPrefillRunner
from argus.tilelang_pypi import TileLangPrefillRunner
from argus.triton_runners import OpenAITritonRunner, TriDaoTritonRunner


def runners(operator, *, causal: bool, gqa: bool, flex_max_autotune: bool = True):
    mask = ops.causal_mask_cond_4d if causal else None
    return [
        FlexAttentionRunner(
            operator, mask_expr=mask, enable_gqa=gqa, max_autotune=flex_max_autotune
        ),
        TorchSDPARunner(operator, AttentionAlgo.FLASH, causal, is_gqa=gqa),
        TorchSDPARunner(operator, AttentionAlgo.CUDNN, causal, is_gqa=gqa),
        OpenAITritonRunner(operator, is_causal=causal),
        TriDaoTritonRunner(operator, is_causal=causal),
        TriDaoRunner(operator, causal=causal),
        FlashInferRunner(operator, is_causal=causal, is_decode=False),
        TileLangPrefillRunner(operator, is_causal=causal, enable_gqa=gqa),
        HelionRunner(operator),
        TawaPrefillRunner(operator, is_causal=causal),
        NeptuneTileLangPrefillRunnerBase(operator, is_causal=causal, enable_gqa=gqa),
    ]


def build_runners(flex_max_autotune: bool = True):
    return {
        op: runners(op, causal=causal, gqa=gqa, flex_max_autotune=flex_max_autotune)
        for op, causal, gqa in (
            (ops.PF_GLOBAL, False, False),
            (ops.PF_CAUSAL, True, False),
            (ops.PF_GQA, True, True),
        )
    }


# bench_new rebuilds this factory for every shape, including profiler children.
bench.build_runners = build_runners
bench.RUNNERS = build_runners()
bench.RUNNERS_BY_OP_NAME = bench.build_runners_by_op_name()
bench.DEFAULT_OPERATOR_SUITE = list(bench.RUNNERS_BY_OP_NAME)
# profiler relaunches this module; preserve the custom registry in the child.
bench.__spec__ = __spec__


if __name__ == "__main__":
    if sys.argv[1:] == ["--list-runners"]:
        print(json.dumps({
            op: [(r.provider_name, r.impl_name) for r in runners_]
            for op, (_, runners_) in bench.build_runners_by_op_name().items()
        }))
    else:
        bench.main()
