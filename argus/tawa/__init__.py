from logging import getLogger
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from ..runner import BenchmarkOperator, BenchmarkRunner

logger = getLogger(__name__)


cached_desc_q = None
cached_desc_k = None
cached_desc_v = None
cached_desc_o = None
cached_desc_m = None
cached_args = None


def get_fwd_tma_descriptors(*args, Osplit=1):
    global cached_desc_q, cached_desc_k, cached_desc_v, cached_desc_o, cached_desc_m, cached_args

    if True:
        cached_args = args
        (
            q, k, v, o, m,
            Z, H, N_CTX, HEAD_DIM,
            BLOCK_M, BLOCK_N,
            qkvo_element_size, m_element_size,
            fp8_v,
        ) = args

        cached_desc_q = TensorDescriptor(
            q,
            [Z * H * N_CTX, HEAD_DIM],
            [HEAD_DIM, 1],
            [BLOCK_M, HEAD_DIM],
        )
        cached_desc_k = TensorDescriptor(
            k,
            [Z * H * N_CTX, HEAD_DIM],
            [HEAD_DIM, 1],
            [BLOCK_N, HEAD_DIM],
        )

        if fp8_v:
            cached_desc_v = TensorDescriptor(
                v, [Z * H * HEAD_DIM, N_CTX], [N_CTX, 1], [HEAD_DIM, BLOCK_N]
            )
        else:
            cached_desc_v = TensorDescriptor(
                v, [Z * H * N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_N, HEAD_DIM]
            )

        cached_desc_o = TensorDescriptor(
            o, [Z * H * N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_M, HEAD_DIM // Osplit]
        )
        cached_desc_m = TensorDescriptor(
            m, [Z * H * N_CTX,], [1,], [BLOCK_M,],
        )

    return cached_desc_q, cached_desc_k, cached_desc_v, cached_desc_o, cached_desc_m


@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i,
    q,
    k_desc_ptr, v_desc_ptr,
    start_m, qk_scale,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr, offs_n: tl.constexpr,
    offs_hz: tl.constexpr, N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX

    offs_kv = offs_hz * N_CTX + lo

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = k_desc_ptr.load([offs_kv, 0])
        qk = tl.dot(q, k.T)

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]

        if fp8_v:
            v = v_desc_ptr.load([offs_hz * HEAD_DIM, start_n])
            p = p.to(tl.float8e5)
            acc = tl.dot(p, v.T, acc)
        else:
            v = v_desc_ptr.load([offs_kv, 0])
            p = p.to(tl.float16)
            acc = tl.dot(p, v, acc)

        m_i = m_ij
        offs_kv += BLOCK_N

    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    Q, K, V,
    q_desc_ptr, k_desc_ptr, v_desc_ptr, o_desc_ptr, m_desc_ptr,
    sm_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504  # 1/log(2)

    q = q_desc_ptr.load([off_hz * N_CTX + start_m * BLOCK_M, 0])

    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            k_desc_ptr, v_desc_ptr,
            start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            4 - STAGE,
            offs_m, offs_n, off_hz, N_CTX,
            V.dtype.element_ty == tl.float8e5,
        )

    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            k_desc_ptr, v_desc_ptr,
            start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            2,
            offs_m, offs_n, off_hz, N_CTX,
            V.dtype.element_ty == tl.float8e5,
        )

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def _tawa_attention(
    q, k, v, causal, sm_scale,
    BLOCK_M=128, BLOCK_N=128,
    NUM_STAGES=2, NUM_WARPS=8,
    USE_TTG_WS=False,
    WG_SPEC=None,
    MATH_WG_PIPE=True,
    FORCE_MEMBAR=False,
    NUM_BLOCK_M=1,
    Osplit=1,
):

    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]

    if v.dtype == torch.float8_e5m2:
        HEAD_DIM_V = v.shape[-2]
    else:
        HEAD_DIM_V = v.shape[-1]

    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    o = torch.empty_like(q)
    stage = 3 if causal else 1

    grid = lambda args: (
        triton.cdiv(q.shape[2], args["BLOCK_M"]),
        q.shape[0] * q.shape[1] // NUM_BLOCK_M,
        1,
    )

    M = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2]),
        device=q.device,
        dtype=torch.float32
    )

    Z, H, N_CTX = k.shape[:3]

    if USE_TTG_WS:
        NUM_STAGES = max(1, NUM_STAGES - 1)

    desc_q, desc_k, desc_v, desc_o, desc_m = get_fwd_tma_descriptors(
        q, k, v, o, M,
        Z, H, N_CTX, HEAD_DIM_Q,
        BLOCK_M, BLOCK_N,
        q.element_size(), M.element_size(),
        v.dtype == torch.float8_e5m2,
        Osplit=Osplit,
    )

    if WG_SPEC == "mma_first":
        WG_SPEC = (("tma_load", NUM_WARPS, 4), ("mma", 0, NUM_WARPS))
    elif WG_SPEC == "tma_load_first":
        WG_SPEC = (("tma_load", 0, 4), ("mma", 4, NUM_WARPS))
    else:
        WG_SPEC = ()

    _attn_fwd[grid](
        q, k, v,
        desc_q, desc_k, desc_v, desc_o, desc_m,
        sm_scale, M, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H,
        N_CTX=q.shape[2],
        HEAD_DIM=HEAD_DIM_K,
        STAGE=stage,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=NUM_STAGES,
        num_warps=NUM_WARPS,
        mma_depth=1,
        enable_warp_specialization=True,
        math_wg_pipe=MATH_WG_PIPE,
        use_ttg_ws=USE_TTG_WS,
        wg_spec_override=WG_SPEC,
        force_membar=FORCE_MEMBAR,
    )

    return o


@dataclass
class TawaPrefillRunner(BenchmarkRunner):

    def __init__(self, operator: BenchmarkOperator, is_causal: bool):
        super().__init__(
            operator,
            provider_name="tawa",
            language="triton",
            impl_name="tma"
        )
        self.is_causal = is_causal
        self.q: torch.Tensor | None = None
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self.gqa = False

        self.num_warps = 8
        self.math_wg_pipe = True
        self.use_ttg_ws = False
        self.wg_spec = None
        self.force_membar = False

    def setup(self, *inputs) -> bool:
        q, k, v = inputs
        self.q, self.k, self.v = q, k, v
        self.gqa = k.shape[1] != q.shape[1]

        if torch.cuda.is_available():
            compute_cap = torch.cuda.get_device_capability()[0]

            if compute_cap >= 10:
                self.num_warps = 4
                self.math_wg_pipe = False
                logger.info("Detected Blackwell GPU: using 4 warps, math_wg_pipe=False")
            elif compute_cap >= 9:
                self.num_warps = 8
                self.math_wg_pipe = True
                logger.info("Detected Hopper GPU: using 8 warps, math_wg_pipe=True")
            else:
                logger.warning(f"GPU compute capability {compute_cap} < 9.0. TAWA attention may not be fully optimized.")
                self.num_warps = 4
                self.math_wg_pipe = False

        return True

    def run(self):
        assert self.q is not None and self.k is not None and self.v is not None

        if self.gqa:
            q, k, v = self._expand_for_gqa(self.q, self.k, self.v)
        else:
            q, k, v = self.q, self.k, self.v

        sm_scale = (q.shape[-1] ** -0.5) if hasattr(self.operator, 'q_heads') else 0.5

        with self.mark_ctx():
            self.output = _tawa_attention(
                q, k, v,
                causal=self.is_causal,
                sm_scale=sm_scale,
                NUM_WARPS=self.num_warps,
                USE_TTG_WS=self.use_ttg_ws,
                WG_SPEC=self.wg_spec,
                MATH_WG_PIPE=self.math_wg_pipe,
                FORCE_MEMBAR=self.force_membar,
            )

    def get_output(self) -> torch.Tensor:
        assert self.output is not None, "Output not set"
        return self.output

    def _expand_for_gqa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        b, q_heads, s, d = q.shape
        _, kv_heads, _, _ = k.shape

        if q_heads == kv_heads:
            return q, k, v

        assert q_heads % kv_heads == 0,
        groups = q_heads // kv_heads

        kv_shape = (b, q_heads, s, d)
        k = k.unsqueeze(2).expand(-1, -1, groups, -1, -1).contiguous().view(kv_shape)
        v = v.unsqueeze(2).expand(-1, -1, groups, -1, -1).contiguous().view(kv_shape)

        return q, k, v
