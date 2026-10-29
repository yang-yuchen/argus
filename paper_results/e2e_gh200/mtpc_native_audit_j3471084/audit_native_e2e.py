"""Audit two timed native-cache forwards; these profiled samples are not performance results."""

from contextlib import contextmanager
import json
from pathlib import Path
import sys

import torch

from scripts import bench_paper_e2e as e2e


prefix = Path(sys.argv[sys.argv.index("--output") + 1])
trace = prefix.with_suffix(".trace.json")
if trace.exists():
    raise FileExistsError(trace)
original_range = torch.cuda.nvtx.range
profiles = []


@contextmanager
def capture(name):
    if not name.startswith("paper_e2e_"):
        with original_range(name):
            yield
        return
    if not profiles:
        profile = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=True,
        )
        profile.__enter__()
        profiles.append(profile)
    with original_range(name), torch.profiler.record_function(name):
        yield


torch.cuda.nvtx.range = capture
e2e.main()
profiles[0].__exit__(None, None, None)
profiles[0].export_chrome_trace(str(trace))
events = json.loads(trace.read_text())["traceEvents"]
copies = [x for x in events if x.get("cat") == "gpu_memcpy"]
large_copies = [x for x in copies if x.get("args", {}).get("bytes", 0) >= 64 * 1024**2]
allocations = [x for x in events if x.get("name") == "[memory]"
               and x.get("args", {}).get("Bytes", 0) >= 64 * 1024**2]
summary = {"trace": str(trace), "profiled_samples_only": True,
           "gpu_memcpy_count": len(copies),
           "large_gpu_copies_ge_64mib": large_copies,
           "large_torch_allocations_ge_64mib": allocations,
           "kernels": sorted({x["name"] for x in events if x.get("cat") == "kernel"})}
prefix.with_suffix(".audit.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({key: value for key, value in summary.items() if key != "kernels"}, indent=2))
