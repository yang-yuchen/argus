import re

import pytest
import torch
import torch._dynamo

from argus.bench import run_operator

# Increase recompile limit for FlexAttention tests
# Running many different kernel configurations (shapes, head counts, etc.)
# in the same process triggers recompilations
torch._dynamo.config.recompile_limit = 128


def result_mismatch(provider: str, impl: str | None = None):
    if impl is None:
        return rf".*: BenchmarkRunner\(.*, provider={provider}.*\) result mismatch against BenchmarkRunner.*"
    else:
        return rf".*: BenchmarkRunner\(.*, provider={provider}, impl={impl}.*\) result mismatch against BenchmarkRunner.*"


def unsupported(provider: str):
    return rf".*: BenchmarkRunner\(.*, provider={provider}.*\) does not support this setup"


class TestOperatorsParameterized:
    @pytest.mark.parametrize(
        "op_name",
        [
            "prefill_global",
            "prefill_causal",
            "prefill_gqa",
            "prefill_alibi",
            "prefill_softcap",
            "prefill_windowed",
            "decode_causal",
            "decode_gqa",
            "decode_alibi",
            "decode_softcap",
        ],
    )
    # test small and large sequence lengths, and test a batch size > 1
    @pytest.mark.parametrize(
        "shape",
        [(1, 128), (1, 2048), (1, 32768), (2, 8192)],
    )
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_operator(self, op_name, shape, caplog: pytest.LogCaptureFixture):
        """Test that operator runs without exceptions."""
        if op_name in {"decode_causal", "decode_gqa"} and shape[0] > 1:
            pytest.xfail("TVM does not support batch > 1 in decoding mode")
        # Capture log messages at WARNING level
        with caplog.at_level("WARNING"):
            run_operator(op_name, shape, n_repeats=3, low_mem_overhead=False)
        exp_warning_ptns = []
        # Tri-Dao Triton impl uses FP16 in part of softmax, which results in worse precision.
        if op_name in {"prefill_global", "prefill_causal", "prefill_gqa"}:
            exp_warning_ptns.append(result_mismatch("triton", "tri-dao"))
        # TODO: We have some result mismatches for Alibi. We should fix it.
        elif op_name == "prefill_alibi" or ((op_name, shape) == ("decode_alibi", (1, 32768))):
            exp_warning_ptns.append(result_mismatch("tri-dao", None))
        # TVM doesn't support super long sequences for prefill.
        if shape[1] >= 8192 and op_name in {"prefill_global", "prefill_causal", "prefill_gqa"}:
            exp_warning_ptns.append(unsupported("tvm"))
        assert len(caplog.records) == len(exp_warning_ptns), (
            f"Expected {len(exp_warning_ptns)} warnings, but got {len(caplog.records)}"
        )
        for record, pattern in zip(caplog.records, exp_warning_ptns):
            assert re.match(pattern, record.message), (
                f"Expected a warning that matches {pattern}, but got {record.message}"
            )


if __name__ == "__main__":
    # Allow running tests directly with: python tests/test_operators.py
    pytest.main([__file__, "-v"])
