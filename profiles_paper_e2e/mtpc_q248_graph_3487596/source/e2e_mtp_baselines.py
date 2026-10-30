"""The TileOPs and FlexAttention APIs used by the official decode/MTP comparison."""

import torch


TILEOPS_COMMIT = "85e7e644f1d99037e937378b8d8dc70c6568113f"


def build_mtp_baseline(implementation, inputs):
    """Bind fixed BHSD input buffers, refreshed by the model before every call."""
    q, k, v = inputs
    if implementation in ("tileops", "tileops_bound"):
        if implementation == "tileops_bound" and q.shape[2] != 1:
            raise ValueError("Bound TileOPs integration only supports single-token decode")
        from tileops.attention import GQADenseFwdOp
        from tileops.backend import BUILTIN

        staged = [x.transpose(1, 2).contiguous() for x in inputs]
        refresh = [(destination, source) for destination, source in zip(staged, inputs)
                   if destination.data_ptr() != source.data_ptr()]
        op = GQADenseFwdOp(is_causal=True, target=BUILTIN)

        def run():
            for destination, source in refresh:
                destination.copy_(source.transpose(1, 2))
            return op(*staged).transpose(1, 2)

        public_output = run()  # Resolve public API kernel selection before model timing.
        kernels = list(op.iter_kernels())
        source = {
            "repository": "https://github.com/tile-ai/TileOPs",
            "commit": TILEOPS_COMMIT,
            "api": "GQADenseFwdOp(is_causal=True,target=BUILTIN)",
            "layout": "BSHD",
            "semantic_mapping": "Hq=Hkv gives MHA",
            "selected_kernel": type(kernels[0]).__name__ if kernels else None,
            "kernel_config": op.kernel_config(),
            "shape": {"q": list(staged[0].shape), "kv": list(staged[1].shape)},
            "scope": "native BSHD storage reused; non-native layout refresh and public op "
                     "allocations included in each call",
        }
        if implementation == "tileops_bound":
            from .e2e_tileops_bound import bind_decode

            if len(kernels) != 1:
                raise ValueError("Bound TileOPs decode requires exactly one selected kernel")
            call, binding = bind_decode(kernels[0], staged, public_output.transpose(1, 2))
            bound_output = call().transpose(1, 2)

            def run():
                for destination, original in refresh:
                    destination.copy_(original.transpose(1, 2))
                call()
                return bound_output

            source.update(binding)
        return run, source
    if implementation == "flex":
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        q_len, kv_len = q.shape[-2], k.shape[-2]
        offset = kv_len - q_len

        def lower_right(_b, _h, q_idx, kv_idx):
            return q_idx + offset >= kv_idx

        # A single last-position query sees every KV entry without a block mask.
        options = {} if q_len == 1 else {"block_mask": create_block_mask(
            lower_right, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=q.device
        )}
        call = torch.compile(
            flex_attention, fullgraph=True, mode="max-autotune-no-cudagraphs"
        )

        def run():
            return call(q, k, v, **options)

        return run, {
            "repository": "https://github.com/pytorch/pytorch",
            "api": "torch.nn.attention.flex_attention",
            "mask": None if q_len == 1 else "q_idx + (KV_LEN-Q_LEN) >= kv_idx",
            "configuration": "fullgraph=True, max-autotune-no-cudagraphs, inference_mode",
        }
    raise ValueError(f"Unsupported MTP baseline: {implementation!r}")
