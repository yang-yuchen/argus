import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .operators import BenchmarkOperator
from .utils import flush_cuda_l2_cache, is_amd_gpu


@dataclass
class BenchmarkRunner(ABC):
    """Base class for all benchmark runners. Subclasses should implement the `setup`, `run`,
    and `get_output` methods."""

    operator: BenchmarkOperator
    provider_name: str
    language: str | None = None
    impl_name: str | None = None
    run_number: int | None = None

    @abstractmethod
    def setup(self, *inputs) -> bool:
        """Setup and warmup the operator. Preprocess the inputs, and store them in this runner instance.
        Return False if the operator is not supported."""
        pass

    @abstractmethod
    def run(self):
        """Run the operator. Use `mark_ctx` to mark the minimal region where the kernel is executed.
        Store the output in this runner instance. Leave any post-processing steps (such as layout
        transformation) to `get_output`."""
        pass

    @abstractmethod
    def get_output(self) -> torch.Tensor:
        """Get the output of the operator. Perform necessary transformations on the output."""
        pass

    def repeat(self, n_repeats: int):
        """Repeat the operator `n_repeats` times. Flush the L2 cache before each run.
        Sleep for a multiple of the estimated kernel duration -- this is a heuristic
        to reduce GPU frequency throttling."""
        from time import sleep, time

        for _ in range(n_repeats):
            flush_cuda_l2_cache()
            self._increment_run()
            begin = time()
            self.run()
            torch.cuda.synchronize()
            duration = time() - begin
            sleep(duration * 3)

    def mark_ctx(self):
        """Mark the region where the kernel is executed, so that the profiler can pick it up.
        It uses NVTX on NVIDIA GPUs and ROCm runtime API on AMD GPUs."""
        name = f"{self.provider_name}({self.impl_name})" if self.impl_name else self.provider_name
        name += "[warmup]" if self.run_number is None else f"[{self.run_number}]"
        if is_amd_gpu():
            from .utils import roctx_range

            return roctx_range(name)
        else:
            import nvtx

            color = "blue" if self.run_number is None else "green"
            return nvtx.annotate(name, color=color)

    def free_members_memory(self):
        """Free as much GPU memory as possible by deleting tensors referenced by this runner instance.
        This is an unsafe operation -- you must re-run before using the runner again."""
        for name in list(self.__dict__.keys()):
            if isinstance(self.__dict__[name], torch.Tensor):
                delattr(self, name)
        gc.collect()
        torch.cuda.empty_cache()

    def _increment_run(self):
        if self.run_number is None:
            self.run_number = 0
        else:
            self.run_number += 1

    def __str__(self):
        kwargs = {
            "operator": self.operator.name,
            "provider": self.provider_name,
        }
        if self.impl_name:
            kwargs["impl"] = self.impl_name
        if self.language:
            kwargs["language"] = self.language
        kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        return f"BenchmarkRunner({kwargs_str})"

    __repr__ = __str__


@dataclass
class DirectOutputRunner(BenchmarkRunner):
    """A runner that directly returns the output of the operator. Inherit this class if the operator
    has no output transformation to perform."""

    output: torch.Tensor | None = None

    @abstractmethod
    def run_output(self) -> torch.Tensor:
        pass

    def run(self):
        self.output = self.run_output()

    def get_output(self) -> torch.Tensor:
        assert self.output is not None, "Output not set"
        return self.output
