"""Fixed-prefix KV storage for the paper e2e attention hook only.

``update`` returns private five-dimensional grouped tensors, not HF's usual
BHSD tensors. Ordinary HF attention and ``generate`` cannot consume this cache.
The caller supplies the same explicit positions after the prefix on each run.
"""

import torch
from transformers.cache_utils import Cache, StaticLayer


class NativeLayer(StaticLayer):
    """Keep each group contiguous in the attention backend's native layout."""

    @torch.no_grad()
    def __init__(self, key_prefix, value_prefix, max_length, group_heads, layout):
        if layout not in ("bshd", "bhsd"):
            raise ValueError("layout must be bshd or bhsd")
        if key_prefix.ndim != 4 or value_prefix.shape != key_prefix.shape:
            raise ValueError("prefix K/V must have the same four-dimensional BHSD shape")
        if key_prefix.dtype != value_prefix.dtype or key_prefix.device != value_prefix.device:
            raise ValueError("prefix K/V dtype and device must match")
        batch, heads, prefix_length, dim = key_prefix.shape
        if group_heads <= 0 or heads % group_heads:
            raise ValueError("group_heads must be positive and divide the model heads")
        if max_length < prefix_length or max_length <= 0:
            raise ValueError("max_length must be positive and fit the prefix")
        super().__init__(max_cache_len=max_length)
        self.layout = layout
        self.group_heads = group_heads
        self.prefix_length = prefix_length
        self.sequence_length = prefix_length
        self.max_batch_size, self.num_heads, self.head_dim = batch, heads, dim
        self.dtype, self.device = key_prefix.dtype, key_prefix.device
        self.num_groups = heads // group_heads
        shape = ((self.num_groups, batch, max_length, group_heads, dim)
                 if layout == "bshd" else
                 (self.num_groups, batch, group_heads, max_length, dim))
        self.keys = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.values = torch.zeros_like(self.keys)
        self.sequence_axis = 2 if layout == "bshd" else 3
        positions = torch.arange(prefix_length, device=self.device)
        self.keys.index_copy_(self.sequence_axis, positions, self._native(key_prefix))
        self.values.index_copy_(self.sequence_axis, positions, self._native(value_prefix))
        self.is_initialized = True
        torch._dynamo.mark_static_address(self.keys)
        torch._dynamo.mark_static_address(self.values)
        self._groups_bhsd = tuple(
            (key.transpose(1, 2), value.transpose(1, 2)) if layout == "bshd"
            else (key, value)
            for key, value in zip(self.keys.unbind(0), self.values.unbind(0))
        )

    def _native(self, states):
        grouped = states.unflatten(1, (self.num_groups, self.group_heads))
        return (grouped.permute(1, 0, 3, 2, 4) if self.layout == "bshd"
                else grouped.permute(1, 0, 2, 3, 4))

    def lazy_initialization(self, key_states):
        raise RuntimeError("NativeLayer must be initialized from a prefix")

    def update(self, key_states, value_states, cache_kwargs=None):
        if key_states.ndim != 4 or value_states.shape != key_states.shape:
            raise ValueError("new K/V must have matching BHSD shapes")
        batch, heads, tokens, dim = key_states.shape
        if ((batch, heads, dim) != (self.max_batch_size, self.num_heads, self.head_dim)
                or tokens <= 0 or self.prefix_length + tokens > self.max_cache_len):
            raise ValueError("new K/V shape does not fit the fixed-prefix cache")
        for states in (key_states, value_states):
            if states.dtype != self.dtype or states.device != self.device:
                raise ValueError("new K/V dtype and device must match the cache")
        positions = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        if (not isinstance(positions, torch.Tensor) or positions.shape != (tokens,)
                or positions.dtype != torch.long or positions.device != self.device):
            raise ValueError("cache_position must be a same-device int64 vector "
                             "with one position per token")
        # Explicit positions are prepared by the fixed-prefix harness; only new tokens move.
        self.keys.index_copy_(self.sequence_axis, positions, self._native(key_states))
        self.values.index_copy_(self.sequence_axis, positions, self._native(value_states))
        self.sequence_length = self.prefix_length + tokens
        return self.keys, self.values

    def get_seq_length(self):
        return self.sequence_length

    def groups_bhsd(self):
        """Return stable ``((K_bhsd, V_bhsd), ...)`` views without packing full KV."""
        return self._groups_bhsd


def build_native_cache(config, prefix_cache, max_length, group_heads, layout):
    """Copy an HF prefix once into private backend-native groups for the e2e hook."""
    if len(prefix_cache.layers) != config.num_hidden_layers:
        raise ValueError("prefix cache layer count does not match the model")
    layers = [NativeLayer(layer.keys, layer.values, max_length, group_heads, layout)
              for layer in prefix_cache.layers]
    cache = Cache(layers=layers)
    cache.layout = layout
    cache.group_heads = group_heads
    return cache
