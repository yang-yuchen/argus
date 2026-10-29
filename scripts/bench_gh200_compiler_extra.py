"""Run the additional compiler decode baselines in the modern GH200 environment."""

from __future__ import annotations

import sys

from argus import operators as ops
from argus.helion import HelionRunner
from argus.triton_runners import TriDaoTritonRunner
import scripts.bench_gh200_compare as base


def build_runners(_flex_max_autotune=True):
    return {ops.DC_CAUSAL: [
        HelionRunner(ops.DC_CAUSAL),
        TriDaoTritonRunner(ops.DC_CAUSAL, is_causal=False),
    ]}


base.build_runners = build_runners
base.bench.build_runners = build_runners
base.bench.RUNNERS = build_runners()
base.bench.RUNNERS_BY_OP_NAME = base.bench.build_runners_by_op_name()
base.bench.DEFAULT_OPERATOR_SUITE = list(base.bench.RUNNERS_BY_OP_NAME)
base.bench.run_operator = base.run_operator
base.bench.__spec__ = __spec__


if __name__ == "__main__":
    args = base.bench.parse_args()
    base.RESULT_DIR = args.profile_dir
    args.func(args)
