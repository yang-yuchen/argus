# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def neptune_attention(q: T.Buffer((1, 4, 64, 128), "float16"), k: T.Buffer((1, 4, 256, 128), "float16"), v: T.Buffer((1, 4, 256, 128), "float16"), T_cast: T.Buffer((1, 4, 64, 128), "float16")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True, "tir.tile_expr_form": True})
        # with T.block("root"):
        T_softmax_expsum_rf = T.alloc_buffer((4, 4, 64))
        T_batch_matmul_NN_rf = T.alloc_buffer((4, 4, 64, 128))
        T_softmax_maxelem_rf_local_fragment_next_global = T.alloc_buffer((4, 4, 64))
        for b0_b1_fused in T.thread_binding(4, thread="blockIdx.y", annotations={"pragma_num_warps": 4, "pragma_tile_optimizer": "tilelang", "pragma_tl_tma": 0}):
            for j_0 in T.thread_binding(4, thread="blockIdx.x"):
                with T.block(""):
                    T.reads(q[0:1, b0_b1_fused:b0_b1_fused + 1, 0:64, 0:128], k[0:1, b0_b1_fused:b0_b1_fused + 1, j_0 * 64:j_0 * 64 + 64, 0:128], v[0:1, b0_b1_fused:b0_b1_fused + 1, j_0 * 64:j_0 * 64 + 64, 0:128])
                    T.writes(T_softmax_expsum_rf[j_0:j_0 + 1, b0_b1_fused:b0_b1_fused + 1, 0:64], T_batch_matmul_NN_rf[j_0:j_0 + 1, b0_b1_fused:b0_b1_fused + 1, 0:64, 0:128], T_softmax_maxelem_rf_local_fragment_next_global[j_0:j_0 + 1, b0_b1_fused:b0_b1_fused + 1, 0:64])
                    T_softmax_maxelem_rf_local_fragment_next = T.alloc_buffer((64,), scope="local.fragment")
                    T_softmax_maxelem_rf_local_fragment_prev_1: T.TensorType((64,), "float32", "local.fragment") = T.tile_full("float32", T.float32(-340282346638528859811704183484516925440.0), 64)
                    T_softmax_expsum_rf_local_fragment_1: T.TensorType((64,), "float32", "local.fragment") = T.tile_full("float32", T.float32(0.0), 64)
                    T_batch_matmul_NN_rf_local_fragment_1: T.TensorType((64, 128), "float32", "local.fragment") = T.tile_full("float32", T.float32(0.0), 64, 128)
                    q_shared_dyn_1: T.TensorType((64, 128), "float16", "shared.dyn") = q[0, b0_b1_fused, 0:64, 0:128]
                    @T.for_with_params_body()
                    def loop_fn_j_1(j_1: T.int32, T_softmax_maxelem_rf_local_fragment_next_1: T.TensorType((64,), "float32", "local.fragment"), T_softmax_expsum_rf_local_fragment_2: T.TensorType((64,), "float32", "local.fragment"), T_batch_matmul_NN_rf_local_fragment_2: T.TensorType((64, 128), "float32", "local.fragment"), T_softmax_maxelem_rf_local_fragment_prev_2: T.TensorType((64,), "float32", "local.fragment")):
                        with T.block(""):
                            T.reads(k[0:1, b0_b1_fused:b0_b1_fused + 1, j_0 * 64 + j_1 * 32:j_0 * 64 + j_1 * 32 + 32, 0:128], T_softmax_maxelem_rf_local_fragment_next[0:64], v[0:1, b0_b1_fused:b0_b1_fused + 1, j_0 * 64 + j_1 * 32:j_0 * 64 + j_1 * 32 + 32, 0:128])
                            T.writes(T_softmax_maxelem_rf_local_fragment_next[0:64])
                            k_shared_dyn_1: T.TensorType((32, 128), "float16", "shared.dyn") = k[0, b0_b1_fused, j_0 * 64 + j_1 * 32:j_0 * 64 + j_1 * 32 + 32, 0:128]
                            T_batch_matmul_NT_local_fragment_1: T.TensorType((64, 32), "float32", "local.fragment") = T.tile_dot("float32", q_shared_dyn_1, T.tile_permute(k_shared_dyn_1, 1, 0))
                            T_score_mod_local_fragment_1: T.TensorType((64, 32), "float32", "local.fragment") = T.if_then_else(j_0 * 64 + j_1 * 32 + T.tile_unsqueeze(T.tile_arange("int32", 0, 32), T.bool(True), T.bool(False)) <= T.tile_unsqueeze(T.tile_arange("int32", 0, 64), T.bool(False), T.bool(True)) + 192, T_batch_matmul_NT_local_fragment_1 * T.float32(0.12751743082459868), T.float32(-340282346638528859811704183484516925440.0))
                            T_softmax_maxelem_rf_rf_local_fragment_1: T.TensorType((64,), "float32", "local.fragment") = T.tile_reduce(T_score_mod_local_fragment_1, 1, "max")
                            T_softmax_maxelem_rf_local_fragment_next_2: T.TensorType((64,), "float32", "local.fragment") = T.max(T_softmax_maxelem_rf_local_fragment_prev_2, T_softmax_maxelem_rf_rf_local_fragment_1)
                            T_softmax_exp_local_fragment_1: T.TensorType((64, 32), "float32", "local.fragment") = T.exp2(T_score_mod_local_fragment_1 - T.tile_unsqueeze(T_softmax_maxelem_rf_local_fragment_next_2, T.bool(False), T.bool(True)))
                            T_softmax_expsum_rf_rf_local_fragment_1: T.TensorType((64,), "float32", "local.fragment") = T.tile_reduce(T_softmax_exp_local_fragment_1, 1, "sum")
                            T_softmax_expsum_rf_local_fragment_3: T.TensorType((64,), "float32", "local.fragment") = T_softmax_expsum_rf_local_fragment_2 * T.exp2(T_softmax_maxelem_rf_local_fragment_prev_2 - T_softmax_maxelem_rf_local_fragment_next_2) + T_softmax_expsum_rf_rf_local_fragment_1
                            T_softmax_exp_cast_shared_dyn_1: T.TensorType((64, 32), "float16", "shared.dyn") = T.Cast("float16", T_softmax_exp_local_fragment_1)
                            v_shared_dyn_1: T.TensorType((32, 128), "float16", "shared.dyn") = v[0, b0_b1_fused, j_0 * 64 + j_1 * 32:j_0 * 64 + j_1 * 32 + 32, 0:128]
                            T_batch_matmul_NN_rf_rf_local_fragment_1: T.TensorType((64, 128), "float32", "local.fragment") = T.tile_dot("float32", T_softmax_exp_cast_shared_dyn_1, v_shared_dyn_1)
                            T_batch_matmul_NN_rf_local_fragment_3: T.TensorType((64, 128), "float32", "local.fragment") = T_batch_matmul_NN_rf_local_fragment_2 * T.tile_unsqueeze(T.exp2(T_softmax_maxelem_rf_local_fragment_prev_2 - T_softmax_maxelem_rf_local_fragment_next_2), T.bool(False), T.bool(True)) + T_batch_matmul_NN_rf_rf_local_fragment_1
                            T_softmax_maxelem_rf_local_fragment_prev_3: T.TensorType((64,), "float32", "local.fragment") = T_softmax_maxelem_rf_local_fragment_next_2
                            T.for_yield([T_softmax_maxelem_rf_local_fragment_next_2, T_softmax_expsum_rf_local_fragment_3, T_batch_matmul_NN_rf_local_fragment_3, T_softmax_maxelem_rf_local_fragment_prev_3])

                    T_softmax_maxelem_rf_local_fragment_next_3, T_softmax_expsum_rf_local_fragment_4, T_batch_matmul_NN_rf_local_fragment_4, T_softmax_maxelem_rf_local_fragment_prev_4 = T.for_with_params(loop_fn_j_1, min=0, max=2, inits=(T_softmax_maxelem_rf_local_fragment_next[0:64], T_softmax_expsum_rf_local_fragment_1, T_batch_matmul_NN_rf_local_fragment_1, T_softmax_maxelem_rf_local_fragment_prev_1), annotations={"num_stages": 2})
                    T_softmax_expsum_rf[j_0, b0_b1_fused, 0:64] = T_softmax_expsum_rf_local_fragment_4
                    T_batch_matmul_NN_rf[j_0, b0_b1_fused, 0:64, 0:128] = T_batch_matmul_NN_rf_local_fragment_4
                    T_softmax_maxelem_rf_local_fragment_next_global[j_0, b0_b1_fused, 0:64] = T_softmax_maxelem_rf_local_fragment_next_3
        for ax1_ax2_ax3_fused in T.thread_binding(256, thread="blockIdx.x", annotations={"pragma_num_warps": 4, "pragma_tile_optimizer": "tilelang", "pragma_tl_tma": 0}):
            with T.block(""):
                T.reads(T_softmax_maxelem_rf_local_fragment_next_global[0:4, ax1_ax2_ax3_fused // 64:ax1_ax2_ax3_fused // 64 + 1, ax1_ax2_ax3_fused % 64:ax1_ax2_ax3_fused % 64 + 1], T_softmax_expsum_rf[0:4, ax1_ax2_ax3_fused // 64:ax1_ax2_ax3_fused // 64 + 1, ax1_ax2_ax3_fused % 64:ax1_ax2_ax3_fused % 64 + 1], T_batch_matmul_NN_rf[0:4, ax1_ax2_ax3_fused // 64:ax1_ax2_ax3_fused // 64 + 1, ax1_ax2_ax3_fused % 64:ax1_ax2_ax3_fused % 64 + 1, 0:128])
                T.writes(T_cast[0:1, ax1_ax2_ax3_fused // 64:ax1_ax2_ax3_fused // 64 + 1, ax1_ax2_ax3_fused % 64:ax1_ax2_ax3_fused % 64 + 1, 0:128])
                T_softmax_maxelem_rf_local_fragment_next_global_local_fragment_1: T.TensorType((4,), "float32", "local.fragment") = T_softmax_maxelem_rf_local_fragment_next_global[0:4, ax1_ax2_ax3_fused // 64, ax1_ax2_ax3_fused % 64]
                T_softmax_maxelem_local_fragment_1: T.TensorType((), "float32", "local.fragment") = T.tile_reduce(T_softmax_maxelem_rf_local_fragment_next_global_local_fragment_1, 0, "max")
                T_softmax_expsum_rf_local_fragment_5: T.TensorType((4,), "float32", "local.fragment") = T_softmax_expsum_rf[0:4, ax1_ax2_ax3_fused // 64, ax1_ax2_ax3_fused % 64]
                T_softmax_expsum_local_fragment_1: T.TensorType((), "float32", "local.fragment") = T.tile_reduce(T_softmax_expsum_rf_local_fragment_5 * T.exp2(T_softmax_maxelem_rf_local_fragment_next_global_local_fragment_1 - T.tile_unsqueeze(T_softmax_maxelem_local_fragment_1, T.bool(True))), 0, "sum")
                T_batch_matmul_NN_rf_local_fragment_5: T.TensorType((4, 128), "float32", "local.fragment") = T_batch_matmul_NN_rf[0:4, ax1_ax2_ax3_fused // 64, ax1_ax2_ax3_fused % 64, 0:128]
                T_batch_matmul_NN_local_fragment_1: T.TensorType((128,), "float32", "local.fragment") = T.tile_reduce(T_batch_matmul_NN_rf_local_fragment_5 * T.exp2(T.tile_unsqueeze(T_softmax_maxelem_rf_local_fragment_next_global_local_fragment_1, T.bool(False), T.bool(True)) - T.tile_unsqueeze(T_softmax_maxelem_local_fragment_1, T.bool(True), T.bool(True))), 0, "sum")
                T_softmax_norm_local_fragment_1: T.TensorType((128,), "float32", "local.fragment") = T_batch_matmul_NN_local_fragment_1 / T.tile_unsqueeze(T_softmax_expsum_local_fragment_1, T.bool(True))
                T_cast[0, ax1_ax2_ax3_fused // 64, ax1_ax2_ax3_fused % 64, 0:128] = T.Cast("float16", T_softmax_norm_local_fragment_1)
                T_cast_1: T.TensorType((128,), "float16", "global") = T.Cast("float16", T_softmax_norm_local_fragment_1)
                T.evaluate(0)