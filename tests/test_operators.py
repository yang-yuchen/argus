"""
Test suite for all supported operators in argus.
This test file verifies that all attention operators can run without exceptions."""

import pytest
import torch

from argus.bench import run_operator


class TestOperatorsParameterized:
    @pytest.mark.parametrize(
        "operator_name",
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
    # test extreme sequence lengths, and test a batch size > 1
    @pytest.mark.parametrize(
        "shape",
        [(1, 128), (1, 2048), (1, 32768), (2, 8192)],
    )
    def test_operator(self, operator_name, shape):
        """Test that operator runs without exceptions."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        run_operator(operator_name, shape, n_repeats=3, low_mem_overhead=False)


if __name__ == "__main__":
    # Allow running tests directly with: python tests/test_operators.py
    pytest.main([__file__, "-v"])
