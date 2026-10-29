"""Official HelionDSL attention and direct PyTorch FlexAttention decode check."""

import importlib.util
from pathlib import Path
import sys

import torch

from argus import operators as ops
from argus.runner import BenchmarkRunner
import scripts.bench_gh200_compare as base


OFFICIAL = Path(__file__).resolve().parents[1] / (
    'argus/paper_baselines/vendor/helion_attention_official/build/torch-cuda'
)


def official_helion():
    name = 'helion_hub_attention'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, OFFICIAL / '__init__.py', submodule_search_locations=[str(OFFICIAL)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class OfficialRunner(BenchmarkRunner):
    def __init__(self, name):
        super().__init__(ops.DC_CAUSAL, name)

    def setup(self, q, k, v):
        self.inputs = (q, k, v)
        if self.provider_name == 'helion_official':
            module = official_helion()
            self.call = module.attention
            self.kwargs = {'causal': False, 'out': torch.empty_like(q)}
            self.source = {
                'url': 'https://huggingface.co/kernels/HelionDSL/attention',
                'commit': 'bffcab17baba6d257c084ad797d1c7bb80e5877f',
                'api': 'attention(q, k, v, causal=False, out=out)',
                'configuration': 'unmodified official SM90 AOT heuristic',
            }
        else:
            from torch.nn.attention.flex_attention import flex_attention

            self.call = torch.compile(
                flex_attention, fullgraph=True, mode='max-autotune-no-cudagraphs'
            )
            self.kwargs = {}
            self.source = {
                'url': 'https://github.com/pytorch/pytorch',
                'commit': torch.version.git_version,
                'api': 'torch.nn.attention.flex_attention.flex_attention',
                'configuration': 'fullgraph=True, max-autotune-no-cudagraphs, inference_mode',
                'split_kv': 'automatic official FlexDecoding heuristic',
            }
        return True

    def run(self):
        with self.mark_ctx():
            self.output = self.call(*self.inputs, **self.kwargs)

    def get_output(self):
        return self.output


def build_runners(_flex_max_autotune=True):
    return {ops.DC_CAUSAL: [OfficialRunner('helion_official'), OfficialRunner('flex_official')]}


base.bench.build_runners = build_runners
base.bench.RUNNERS_BY_OP_NAME = base.bench.build_runners_by_op_name()
base.bench.__spec__ = __spec__
base.bench.run_operator = torch.inference_mode()(base.run_operator)


if __name__ == '__main__':
    args = base.bench.parse_args()
    base.RESULT_DIR = args.profile_dir
    base.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    args.func(args)
