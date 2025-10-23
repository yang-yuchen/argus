
from logging import getLogger
from dataclasses import dataclass

import torch
import tilelang as tl
import tilelang.language as T
from tilelang.engine import lower as tl_lower
from tilelang.jit.adapter import TorchDLPackKernelAdapter

from ..runner import BenchmarkOperator, BenchmarkRunner
from tilelang.autotuner import autotune
from tilelang.jit import jit
from .decode import TileLangDecodeRunner

logger = getLogger(__name__)

@dataclass
class TileLangPrefillRunner(BenchmarkRunner):
    def __init__(self, operator: BenchmarkOperator, is_causal: bool, enable_gqa: bool = False):
        super().__init__(operator, provider_name="tilelang", language="tilelang", impl_name=None)
        self.is_causal = is_causal
        self.enable_gqa = enable_gqa
        self.q_bshd: torch.Tensor | None = None
        self.k_bshd: torch.Tensor | None = None
        self.v_bshd: torch.Tensor | None = None
        self.output_bshd: torch.Tensor | None = None
        self._tl_func = None
        self._built_shape_key = None

    def setup(self, *inputs) -> bool:
        q, k, v = inputs
        self.q_bshd = torch.permute(q, (0, 2, 1, 3)).contiguous()
        self.k_bshd = torch.permute(k, (0, 2, 1, 3)).contiguous()
        self.v_bshd = torch.permute(v, (0, 2, 1, 3)).contiguous()
        assert self.q_bshd.shape[0] == self.k_bshd.shape[0] == self.v_bshd.shape[0]
        assert self.q_bshd.shape[-1] == self.k_bshd.shape[-1] == self.v_bshd.shape[-1]
        self._maybe_build_tilelang_kernel()
        return True

    def run(self):
        assert self.q_bshd is not None and self.k_bshd is not None and self.v_bshd is not None
        assert self._tl_func is not None, "TileLang kernel not built"
        with self.mark_ctx():
            self.output_bshd = self._tl_func(self.q_bshd, self.k_bshd, self.v_bshd)

    def get_output(self) -> torch.Tensor:
        assert self.output_bshd is not None, "Output not set"
        return torch.permute(self.output_bshd, (0, 2, 1, 3))

    def _maybe_build_tilelang_kernel(self):
        assert self.q_bshd is not None and self.k_bshd is not None and self.v_bshd is not None
        b, sq, hq, d = self.q_bshd.shape
        _, sk, hk, _ = self.k_bshd.shape
        if sq != sk:
            return
        groups = 1
        if self.enable_gqa and hq != hk:
            assert hq % hk == 0, "Q heads must be multiple of KV heads for GQA"
            groups = hq // hk
        shape_key = (b, sq, hq, hk, d, self.is_causal, groups)
        if self._built_shape_key == shape_key and self._tl_func is not None:
            logger.info("TileLang kernel already built for shape %s, skipping autotuning", shape_key)
            return
        logger.info("Building TileLang kernel for shape %s", shape_key)
        dtype = "float16"
        accum_dtype = "float"
        heads = hq
        head_kv = hk if groups != 1 else hq
        seq_len = sq
        is_causal = self.is_causal

        def build_kernel(block_M: int, block_N: int, num_stages: int, threads: int):
            @T.macro
            def MMA0(
                K: T.Tensor([b, seq_len, head_kv, d], dtype),
                Q_shared: T.SharedBuffer([block_M, d], dtype),
                K_shared: T.SharedBuffer([block_N, d], dtype),
                acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
                k: T.int32,
                bx: T.int32,
                by: T.int32,
                bz: T.int32,
            ):
                T.copy(K[bz, k * block_N:(k + 1) * block_N, by // groups, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0, -T.infinity(accum_dtype))
                else:
                    T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            @T.macro
            def MMA1(
                V: T.Tensor([b, seq_len, head_kv, d], dtype),
                V_shared: T.SharedBuffer([block_N, d], dtype),
                acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
                acc_o: T.FragmentBuffer([block_M, d], accum_dtype),
                k: T.int32,
                by: T.int32,
                bz: T.int32,
            ):
                T.copy(V[bz, k * block_N:(k + 1) * block_N, by // groups, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            scale = (1.0 / d) ** 0.5 * 1.44269504

            @T.macro
            def Softmax(
                acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
                acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
                scores_max: T.FragmentBuffer([block_M], accum_dtype),
                scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
                scores_sum: T.FragmentBuffer([block_M], accum_dtype),
                logsum: T.FragmentBuffer([block_M], accum_dtype),
            ):
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

            @T.macro
            def Rescale(
                acc_o: T.FragmentBuffer([block_M, d], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
            ):
                for i, j in T.Parallel(block_M, d):
                    acc_o[i, j] *= scores_scale[i]

            @T.prim_func
            def main(
                Q: T.Tensor([b, seq_len, heads, d], dtype),
                K: T.Tensor([b, seq_len, head_kv, d], dtype),
                V: T.Tensor([b, seq_len, head_kv, d], dtype),
                Output: T.Tensor([b, seq_len, heads, d], dtype),
            ):
                with T.Kernel(T.ceildiv(seq_len, block_M), heads, b, threads=threads) as (bx, by, bz):
                    Q_shared = T.alloc_shared([block_M, d], dtype)
                    K_shared = T.alloc_shared([block_N, d], dtype)
                    V_shared = T.alloc_shared([block_N, d], dtype)
                    O_shared = T.alloc_shared([block_M, d], dtype)
                    acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                    acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                    acc_o = T.alloc_fragment([block_M, d], accum_dtype)
                    scores_max = T.alloc_fragment([block_M], accum_dtype)
                    scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                    scores_scale = T.alloc_fragment([block_M], accum_dtype)
                    scores_sum = T.alloc_fragment([block_M], accum_dtype)
                    logsum = T.alloc_fragment([block_M], accum_dtype)

                    T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                    T.fill(acc_o, 0)
                    T.fill(logsum, 0)
                    T.fill(scores_max, -T.infinity(accum_dtype))

                    loop_range = (
                        T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N))
                        if is_causal else T.ceildiv(seq_len, block_N)
                    )

                    for k in T.Pipelined(
                            loop_range,
                            num_stages=num_stages,
                            order=[-1, 0, 3, 1, -1, 2],
                            stage=[-1, 0, 0, 1, -1, 1],
                            group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10], [11], [12], [13]]):
                        MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                        Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                        Rescale(acc_o, scores_scale)
                        MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
                    for i, j in T.Parallel(block_M, d):
                        acc_o[i, j] /= logsum[i]
                    T.copy(acc_o, O_shared)
                    T.copy(O_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])

            return main

        # try:
        

        dtype_size = torch.tensor([], dtype=getattr(torch, dtype)).element_size()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            max_shmem_bytes = getattr(props, "shared_memory_per_block_optin", 0)
            max_shmem_bytes = max(max_shmem_bytes, props.shared_memory_per_block)
        else:
            max_shmem_bytes = 256 * 1024

        def _estimated_shared_mem_bytes(cfg):
            stage_count = max(cfg["num_stages"], 1)
            block_m = cfg["block_M"]
            block_n = cfg["block_N"]
            shared_q = block_m * d
            shared_o = block_m * d
            shared_kv = 2 * stage_count * block_n * d
            shared_elems = shared_q + shared_o + shared_kv
            return (shared_elems + 8 * d) * dtype_size

        def _valid(cfg):
            warp_cnt = cfg["threads"] // 32
            if warp_cnt <= 0 or (cfg["block_M"] % (16 * warp_cnt) != 0):
                return False
            if max_shmem_bytes and _estimated_shared_mem_bytes(cfg) > max_shmem_bytes:
                return False
            return True

        def get_configs():
            block_M = [64, 128, 256]
            block_N = [64, 128, 256]
            num_stages = [0, 1, 2, 3]
            threads = [128, 256]
            configs = [
                {"block_M": bm, "block_N": bn, "num_stages": ns, "threads": th}
                for bm in block_M
                for bn in block_N
                for ns in num_stages
                for th in threads
            ]
            return [cfg for cfg in configs if _valid(cfg)]

        import uuid
        unique_id = str(uuid.uuid4())[:8]
        
        @autotune(
            configs=get_configs(),
            warmup=3,
            rep=5,
        )
        @jit(
            out_idx=[3],
            pass_configs={
                tl.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            }
        )
        def flashattn_kernel(block_M=64, block_N=64, num_stages=1, threads=128, _cache_key=unique_id):
            return build_kernel(block_M, block_N, num_stages, threads)

        best_result = flashattn_kernel()
        logger.info("TileLang tuned best config: %s (%.3f ms)", best_result.config, best_result.latency * 1e3)
        
        self._tl_func = best_result
        self._built_shape_key = shape_key