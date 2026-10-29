"""Measure audited official FP8 attention with the existing paper nsys protocol."""

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform

import torch

from argus.operators import BenchmarkOperator
from argus.paper_baselines import fp8
from argus.runner import BenchmarkRunner
from argus.utils import profiler_ctx, relaunch_with_profiler
from scripts.bench_paper_sweep import collect_profile, digest, load_inputs, reference, summarize_profile


class Runner(BenchmarkRunner):
    def __init__(self, case, method):
        super().__init__(BenchmarkOperator(case), f'fp8_{method}', method, case)
        self.case, self.method = case, method

    def setup(self, *inputs):
        module = importlib.import_module(f'argus.paper_baselines.fp8_{self.method}')
        kind = fp8.spec(self.case)[0].removesuffix('8')
        self.prepared = module.prepare(*fp8.canonical_inputs(inputs),
                                       causal=kind in ('prefillc', 'mtpc'))
        self.run()
        return True

    def run(self):
        with self.mark_ctx():
            self.output = self.prepared.run()

    def get_output(self):
        return self.output


@torch.no_grad()
def measure(case, method, output, repeats, input_file):
    if not torch.cuda.is_available() or platform.machine() != 'aarch64':
        raise RuntimeError('Run on an allocated aarch64 GH200 GPU')
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = load_inputs(case, input_file)
    expected = reference(case, inputs)
    runner = Runner(case, method)
    runner.setup(*inputs)
    for _ in range(2):
        runner.run()
        checks = fp8.check_output(runner.get_output(), expected)
    metadata = {'case': case, 'implementation': f'fp8_{method}',
                'display_name': 'TileLang' if method == 'tilelang' else 'FlexAttn',
                **runner.prepared.metadata, 'fp8_contract': fp8.contract(),
                'correct': True, **checks, 'repeats': repeats,
                'inputs_sha256': digest(inputs), 'input_file': str(input_file),
                'host': platform.node(), 'gpu': torch.cuda.get_device_name(),
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'runtime_versions': {name: importlib.metadata.version(name) for name in
                                     ('tilelang', 'apache-tvm-ffi', 'triton')},
                'metric': 'sum of GPU kernel durations per cold-L2 NVTX repeat (us)',
                'l2_flush_bytes': 200 * 1024**2, 'warmup': 10,
                'measurement_complete': False}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for _ in range(10):
        runner.run()
    torch.cuda.synchronize()
    with profiler_ctx():
        runner.repeat(repeats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case')
    parser.add_argument('--method', choices=('tilelang', 'flex'), default='tilelang')
    parser.add_argument('--input-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeat', type=int, default=50)
    parser.add_argument('--profile-child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        fp8.spec(args.case)
    except ValueError as error:
        parser.error(str(error))
    if args.repeat < 1 or not args.input_file.is_file():
        parser.error('Require positive repeats and an existing shared input file')
    if args.method == 'flex':
        from argus.paper_baselines.fp8_flex import UNSUPPORTED_REASON
        parser.error(UNSUPPORTED_REASON)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.with_suffix('.json').exists() or output.with_suffix('.nsys-rep').exists():
        parser.error('Output exists; choose a new --output prefix')
    if args.profile_child:
        measure(args.case, args.method, output, args.repeat, args.input_file.resolve())
        return
    command = ['-m', 'scripts.bench_fp8_baseline', args.case, '--method', args.method,
               '--input-file', str(args.input_file.resolve()), '--output', str(output),
               '--repeat', str(args.repeat), '--profile-child']
    result = collect_profile(
        output,
        lambda: relaunch_with_profiler(output, 'nsys', False, command,
                                       wait_for_cuda_profiler=True),
        lambda: summarize_profile(output, args.repeat, f'fp8_{args.method}({args.case})'),
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
