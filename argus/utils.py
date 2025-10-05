from contextlib import contextmanager
from functools import cache
from logging import getLogger
from pathlib import Path

import torch

logger = getLogger(__name__)


def relaunch_with_profiler(
    output_prefix: Path,
    profiler: str | None,
    gpu_metrics: bool,
    relaunch_args: list[str],
    nsys_wait: bool = True,
):
    import sys
    import time
    from subprocess import check_call
    from tempfile import TemporaryDirectory

    # "-u" to unbuffer stdout, so that we can see the progress of the run.
    relaunch_args = [sys.executable, "-u"] + relaunch_args
    if profiler == "ncu":
        if is_amd_gpu():
            raise ValueError("ncu profiler does not support AMD GPUs")
        if gpu_metrics:
            raise ValueError("ncu profiler does not support gpu-metrics")
        if output_prefix.with_suffix(".ncu-rep").exists():
            logger.info(f"Skipping {output_prefix}.ncu-rep because it already exists")
            return
        logger.info(f"Running with ncu; saving profile to {output_prefix}.ncu-rep")
        ncu_args = ["ncu", "--nvtx", "--set", "full", "-o", output_prefix]
        ncu_args += [
            "--profile-from-start",
            "off",
        ]  # Profile only after cudaProfilerStart().
        check_call(ncu_args + relaunch_args)
    elif profiler == "nsys":
        if is_amd_gpu():
            raise ValueError("ncu profiler does not support AMD GPUs")
        if output_prefix.with_suffix(".nsys-rep").exists():
            logger.info(f"Skipping {output_prefix}.nsys-rep because it already exists")
            return
        logger.info(f"Running with nsys; saving profile to {output_prefix}.nsys-rep")
        # Select cuda, nvtx, and osrt traces.
        nsys_args = ["nsys", "profile", "-o", output_prefix, "--trace=cuda,nvtx,osrt"]
        if gpu_metrics:
            nsys_args += ["--gpu-metrics-devices=cuda-visible"]
        if nsys_wait:
            # Let NSYS sync with the main process of the workload
            # and return with the same exit code as that process.
            # Does not work well with full-DNN inference workloads because PyTorch does... something.
            nsys_args += ["--wait=primary"]
        check_call(nsys_args + relaunch_args)
    elif profiler == "rocprof":
        if not is_amd_gpu():
            raise ValueError("rocprof profiler does not support NVIDIA GPUs")
        if gpu_metrics:
            raise ValueError("rocprof profiler does not support gpu-metrics")
        if output_prefix.with_suffix(".pftrace").exists():
            logger.info(f"Skipping {output_prefix}.pftrace because it already exists")
            return
        # rocprofv3 wants to write the profile result of every process to a separate file,
        # while we want a single file. It's the best if we tell it to write to a temp dir,
        # then concatenate the files and move it to where we want.
        with TemporaryDirectory() as tmpdir:
            roctx_args = [
                "rocprofv3",
                "--runtime-trace",
                "--kernel-trace",
                "--kernel-rename",
            ]
            roctx_args += ["--output-format", "pftrace", "-d", tmpdir]
            check_call(roctx_args + ["--"] + relaunch_args)
            # Believe it or not, `rocprofv3` has an async component that we don't know when it's done.
            # So we have to wait for it to finish.
            time.sleep(10)
            # Concatenate the files.
            with open(output_prefix.with_suffix(".pftrace"), "wb") as fout:
                # Use rglob. rocprofv3 likes to create another subdir (with our hostname).
                for file in Path(tmpdir).rglob("**/*.pftrace"):
                    fout.write(file.read_bytes())
    else:
        raise ValueError(f"Unknown profiler: {profiler}")


def flush_cuda_l2_cache(n_bytes: int = 200 * 1024**2):
    buf = torch.empty(n_bytes, dtype=torch.uint8, device="cuda")
    buf.zero_()
    torch.cuda.synchronize()


@cache
def is_amd_gpu():
    return torch.cuda.get_device_name(0).startswith("AMD")


@contextmanager
def profiler_ctx():
    use_cuda_profiler = not is_amd_gpu()
    if use_cuda_profiler:
        torch.cuda.cudart().cudaProfilerStart()  # type: ignore
        yield
        torch.cuda.cudart().cudaProfilerStop()  # type: ignore
    else:
        # Don't do anything for AMD. hipProfilerStart() has been deprecated. We'll use roctx instead.
        yield


@cache
def get_roctx():
    import ctypes

    lib = ctypes.cdll.LoadLibrary("/opt/rocm/lib/librocprofiler-sdk-roctx.so")
    # Signatures
    lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
    lib.roctxRangePushA.restype = ctypes.c_int
    lib.roctxRangePop.argtypes = []
    lib.roctxRangePop.restype = ctypes.c_int
    return lib


@contextmanager
def roctx_range(msg):
    lib = get_roctx()
    rid = lib.roctxRangePushA(msg.encode("utf-8"))
    if rid < 0:
        raise RuntimeError("roctxRangePushA failed")
    try:
        yield
    finally:
        rc = lib.roctxRangePop()
        if rc != 0:
            raise RuntimeError(f"roctxRangePop returned {rc}")


def bnsh_expand_for_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    q_heads = q.shape[1]
    b, kv_heads, s, d = k.shape
    groups = q_heads // kv_heads
    kv_shape = (b, q_heads, s, d)
    k = k.unsqueeze(2).expand(-1, -1, groups, -1, -1).contiguous().view(kv_shape)
    v = v.unsqueeze(2).expand(-1, -1, groups, -1, -1).contiguous().view(kv_shape)
    return q, k, v
