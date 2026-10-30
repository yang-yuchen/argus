"""Shared FLOPs accounting for the existing Llama-7B E2E measurements."""


def model_flops(length, query_length=16):
    """Original full-model convention with lower-right causal cached attention."""
    if not 0 < query_length <= length:
        raise ValueError("Query length must be positive and fit the visible KV length")
    q, h, intermediate, heads, dim, layers = query_length, 4096, 11008, 32, 128, 32
    projections = 2 * q * h * (3 * heads * dim) + 2 * q * (heads * dim) * h
    mlp = 2 * q * (h * intermediate + intermediate * h)
    pairs = q * length - q * (q - 1) // 2
    return layers * (projections + mlp + 4 * heads * dim * pairs)
