from logging import getLogger

import numpy as np
import torch

from ..runner import BenchmarkOperator, BenchmarkRunner

logger = getLogger(__name__)


class TVMPrefillRunner(BenchmarkRunner):
    try:
        import tvm
    except ImportError as e:
        tvm = e

    def __init__(self, operator: BenchmarkOperator, is_causal: bool):
        super().__init__(operator, "tvm", language="tvm")
        self.is_causal = is_causal

    def setup(self, *inputs) -> bool:
        if isinstance(self.tvm, ImportError):
            logger.warning("tvm import failed with exception: %s", self.tvm)
            return False

        import tvm
        from tvm import relax

        from .get_attn import get_attention_module

        q, k, v = inputs
        # Plain Attention cannot handle too large inputs, because the score tensor gets too large.
        score_shape = (q.shape[0], q.shape[1], q.shape[2], k.shape[2])
        if np.prod(score_shape) >= 2**32:  # 1 billion elements, 4GB
            return False
        mod = get_attention_module(q.shape, k.shape, v.shape, self.is_causal)
        device = tvm.device("cuda", 0)
        target = tvm.target.Target.from_device(device)
        with tvm.transform.PassContext(config={"cuda.kernels_output_dir": "/tmp/tvm_kernels"}):
            exe = relax.build(mod, target, relax_pipeline=relax.get_pipeline("opt_llm"))
        self.vm = relax.VirtualMachine(exe, device)
        # TODO: tvm.nd.array(torch_array, device) segfaults for arrays that are roughly larger than 1GB.
        self.q, self.k, self.v = [tvm.nd.array(data.cpu().numpy(), device) for data in [q, k, v]]
        return True

    def run(self):
        with self.mark_ctx():
            self.output = self.vm["main"](self.q, self.k, self.v)

    def get_output(self) -> torch.Tensor:
        return torch.from_numpy(self.output.numpy())


class TVMDecodeRunner(BenchmarkRunner):
    try:
        import tvm
    except ImportError as e:
        tvm = e

    def __init__(self, operator: BenchmarkOperator):
        super().__init__(operator, "tvm", language="tvm")

    def setup(self, *inputs) -> bool:
        if isinstance(self.tvm, ImportError):
            logger.warning("tvm import failed with exception: %s", self.tvm)
            return False

        import tvm
        from tvm import relax

        from .get_attn import get_attention_module

        q, k, v = inputs
        mod = get_attention_module(q.shape, k.shape, v.shape, is_causal=False)
        device = tvm.device("cuda", 0)
        target = tvm.target.Target.from_device(device)
        exe = relax.build(mod, target, relax_pipeline=relax.get_pipeline("opt_llm"))
        self.vm = relax.VirtualMachine(exe, device)
        self.q, self.k, self.v = [tvm.nd.array(data.cpu().numpy(), device) for data in [q, k, v]]
        return True

    def run(self):
        with self.mark_ctx():
            self.output = self.vm["main"](self.q, self.k, self.v)

    def get_output(self) -> torch.Tensor:
        return torch.from_numpy(self.output.numpy())
