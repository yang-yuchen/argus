def causal_topk_indices(num_tokens: int, topk: int, seed: int = 0, chunk: int = 1024):
    """[num_tokens, topk] int32 (CPU): for token t, min(t + 1, topk) distinct random positions in [0, t]
    in random order, then -1. Generated on the GPU in chunks of tokens (random scores, masked to
    the causal prefix, top-k), so a 32K-token sweep takes seconds."""
    import torch

    gen = torch.Generator(device="cuda").manual_seed(seed)
    pos = torch.arange(num_tokens, device="cuda")
    out = torch.empty((num_tokens, topk), dtype=torch.int32)
    for t0 in range(0, num_tokens, chunk):
        t1 = min(num_tokens, t0 + chunk)
        scores = torch.rand((t1 - t0, num_tokens), generator=gen, device="cuda")
        scores.masked_fill_(pos[None, :] > pos[t0:t1, None], -1.0)
        vals, idx = scores.topk(topk, dim=1)
        idx = idx.to(torch.int32)
        idx[vals < 0] = -1
        out[t0:t1] = idx.cpu()
    return out
