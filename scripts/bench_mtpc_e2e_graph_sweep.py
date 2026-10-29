"""Run one MTP backend's whole-step graph replay over all paper KV lengths."""

import argparse
import gc
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.bench_paper_e2e import measure

LENGTHS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("implementation", choices=("tileops", "flex", "nautilus"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 1:
        parser.error("--repeat and --warmup must be positive")
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    outputs = [directory / f"mtpc_L{length}_{args.implementation}" for length in LENGTHS]
    if any(output.with_suffix(".json").exists() for output in outputs):
        parser.error("An output already exists; use a new directory")
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    for length, output in zip(LENGTHS, outputs):
        print(f"Starting MTP L{length}/{args.implementation}", flush=True)
        try:
            measure(SimpleNamespace(variant="mtpc", implementation=args.implementation,
                                    kv_length=length, output=output, repeat=args.repeat,
                                    warmup=args.warmup, cudagraph=True, profile_child=False))
        finally:
            ALL_ATTENTION_FUNCTIONS.register("paper_e2e", ALL_ATTENTION_FUNCTIONS["sdpa"])
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
        print(f"Finished MTP L{length}/{args.implementation}", flush=True)


if __name__ == "__main__":
    main()
