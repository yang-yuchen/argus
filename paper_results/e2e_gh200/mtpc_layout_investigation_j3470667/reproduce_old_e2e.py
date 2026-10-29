"""Controlled reproduction of the old TileOPs double-copy integration on the same GPU."""

import torch

from argus.paper_baselines.e2e_mtp_baselines import build_mtp_baseline
from scripts import bench_paper_e2e as e2e


class OldTileOpsCore(e2e.AttentionCore):
    def __init__(self, variant, inputs, implementation, kv_length=None):
        assert variant == "mtpc" and implementation == "tileops"
        super().__init__(variant, inputs, implementation, kv_length)
        self.buffers = [torch.empty_like(x, memory_format=torch.contiguous_format) for x in inputs]
        for destination, source in zip(self.buffers, inputs):
            destination.copy_(source)
        self.call, self.source = build_mtp_baseline("tileops", self.buffers)
        self.source["diagnostic"] = "old BHSD staging followed by native BSHD refresh"


e2e.AttentionCore = OldTileOpsCore
e2e.main()
