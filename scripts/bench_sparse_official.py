"""Measure official compiler sparse attention on saved Nautilus sweep inputs."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch

from argus.paper_baselines.strong_validation import accuracy, reference_fp64
from argus.runner import BenchmarkRunner
from argus.operators import BenchmarkOperator
from argus.utils import profiler_ctx, relaunch_with_profiler
from scripts.bench_paper_sweep import digest, summarize_profile

ROOT = Path(__file__).resolve().parents[1]


class Runner(BenchmarkRunner):
    def __init__(self, case, method, call):
        super().__init__(BenchmarkOperator(case), "sparse_official", "compiler", method)
        self.call = call

    def setup(self, *inputs):
        return True

    def run(self):
        with self.mark_ctx():
            self.output = self.call()

    def get_output(self):
        return self.output


def build(method, inputs):
    if method == "tilelang":
        from argus.paper_baselines.sparse_tilelang_example_official import build as factory
    elif method == "flex":
        from argus.paper_baselines.sparse_flex_official import build as factory
    elif method == "helion":
        from argus.paper_baselines.sparse_flex_official import prepare
        from argus.paper_baselines.sparse_helion_official import build_prepared
        query, key, value, mask = prepare(*inputs)
        call = build_prepared(query, key, value, mask)

        def run():
            return call().squeeze(0).permute(1, 0, 2)

        run.source = call.source
        return run
    else:
        raise ValueError(method)
    return factory(*inputs)


def measure(args):
    case = f"sparse_L{args.length}"
    path = args.inputs / f"{case}_inputs.pt"
    cpu = torch.load(path, map_location="cpu", weights_only=True)
    length = args.length
    shapes = [(length, 16, 576), (length, 576), (length, min(length, 2048))]
    for tensor, shape, dtype in zip(cpu, shapes, [torch.bfloat16, torch.bfloat16, torch.int32]):
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"Unexpected saved tensor: {tensor.shape}, {tensor.dtype}")
    inputs = [x.cuda().contiguous() for x in cpu]
    inputs_hash = digest(inputs)
    oracle_meta = json.loads((args.inputs / f"{case}.json").read_text())
    if not oracle_meta['correct'] or oracle_meta['inputs_sha256'] != inputs_hash:
        raise ValueError('Nautilus validation or saved input hash mismatch')
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        ref = reference_fp64('sparse', inputs)
        call = build(args.method, inputs)
        runner = Runner(case, args.method, call)
        runner.run()
        torch.cuda.synchronize()
        check = accuracy(ref, runner.get_output(), "sparse")
        if not check['passed']:
            raise AssertionError(json.dumps(check))
        packages = {}
        for name in ('torch', 'triton', 'tilelang', 'helion', 'tileops'):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        metadata = {
            'case': case, 'implementation': args.method, 'source': call.source,
            'correct': True, 'accuracy': check, 'relative_l2': check['relative_l2'],
            'inputs_sha256': inputs_hash, 'repeats': args.repeat,
            'metric': 'sum GPU kernel durations; 200 MiB cold-L2 flush each repeat',
            'measurement_complete': False,
            'environment': {'host': __import__('platform').node(),
                            'gpu': torch.cuda.get_device_name(),
                            'python': __import__('sys').executable,
                            'packages': packages},
        }
        args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
        for _ in range(10):
            runner.run()
        torch.cuda.synchronize()
        with profiler_ctx():
            runner.repeat(args.repeat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=['tilelang', 'flex', 'helion'])
    parser.add_argument('length', type=int, choices=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeat', type=int, default=50)
    parser.add_argument('--profile-child', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.inputs = args.inputs.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.profile_child:
        measure(args)
        return
    relaunch_with_profiler(
        args.output, 'nsys', False,
        ['-m', 'scripts.bench_sparse_official', args.method, str(args.length),
         '--inputs', str(args.inputs), '--output', str(args.output),
         '--repeat', str(args.repeat), '--profile-child'],
        wait_for_cuda_profiler=True,
    )
    summarize_profile(args.output, args.repeat, f'sparse_official({args.method})')


if __name__ == '__main__':
    main()
