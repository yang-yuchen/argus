# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def neptune_attention(q: T.Buffer((1, 4, 8, 128), "float16"), k: T.Buffer((1, 4, 16384, 128), "float16"), v: T.Buffer((1, 4, 16384, 128), "float16"), T_cast: T.Buffer((1, 4, 8, 128), "float16")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True, "tir.tile_expr_form": True})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"nautilus_applied_rules": "tile(T_batch_matmul_NT, SRSR); split-kv(T_batch_matmul_NT); RCA(T_score_mod, j_0_1); SKV(T_softmax_maxelem, j_0_0); RCA(T_softmax_exp_cast, i3); SKV(T_batch_matmul_NN, j_0_0); SKV(T_softmax_expsum, j_0_0); RCA(T_softmax_norm, ax1_ax2_ax3_fused); RCA(T_cast, ax1_ax2_ax3_fused); RCA(T_cast_crop, ax1_ax2_ax3_fused); ", "schedule_nums": "1,2,3,4,5,6,7,8,9,10,30,31,32,33,34,35,36,37,38,39,44,108,110,111,112,114,116,117,118,119,908,909,910,911,912,913,914,915,916,917,1138,1139,1140,1141,1142,1143,1144,1145,1146,1147,2168,2169,2170,2171,2172,2173,2174,2175,2176,2177,2398,2399,2400,2401,2402,2403,2404,2405,2406,2407,"})
            T_batch_matmul_NN_rf = T.alloc_buffer((32, 4, 16, 128))
            T_softmax_expsum_rf = T.alloc_buffer((32, 4, 16))
            T_softmax_maxelem_rf_next_global = T.alloc_buffer((32, 4, 16))
            for b0 in T.thread_binding(1, thread="blockIdx.z", annotations={"pragma_nautilus_fold_accumulators": 1, "pragma_num_warps": 4, "pragma_tile_optimizer": "tilelang", "pragma_tl_num_stages": 1, "pragma_tl_tma": 1, "pragma_tl_warp_specialize": 1, "pragma_trcg_pointer_method": 3, "pragma_triton_enable_warp_specialization": T.bool(True), "pragma_triton_force_membar": T.bool(False), "pragma_triton_math_wg_pipe": T.bool(True), "pragma_triton_mma_depth": 1, "pragma_triton_num_stages": 2, "pragma_triton_num_warps": 4, "pragma_triton_use_ttg_ws": T.bool(False)}):
                for b1 in T.thread_binding(4, thread="blockIdx.y"):
                    for i_0 in T.serial(1, annotations={"nautilus_outer_tiled_loop": T.bool(True)}):
                        for j_0_0 in T.thread_binding(32, thread="blockIdx.x", annotations={"nautilus_fusion_role": "split", "nautilus_outer_tiled_loop": T.bool(True), "num_stages": 1}):
                            with T.block(""):
                                T.reads(q[0:1, b1:b1 + 1, 0:16, 0:128], k[0:1, b1:b1 + 1, j_0_0 * 512:j_0_0 * 512 + 512, 0:128], v[0:1, b1:b1 + 1, j_0_0 * 512:j_0_0 * 512 + 512, 0:128])
                                T.writes(T_batch_matmul_NN_rf[j_0_0:j_0_0 + 1, b1:b1 + 1, 0:16, 0:128], T_softmax_expsum_rf[j_0_0:j_0_0 + 1, b1:b1 + 1, 0:16], T_softmax_maxelem_rf_next_global[j_0_0:j_0_0 + 1, b1:b1 + 1, 0:16])
                                T_softmax_maxelem_rf_next_local_fragment = T.alloc_buffer((16,), scope="local.fragment")
                                T_softmax_maxelem_rf_prev_local_fragment_1: T.TensorType((16,), "float32", "local.fragment") = T.tile_full("float32", T.float32(-340282346638528859811704183484516925440.0), 16)
                                T_batch_matmul_NN_rf_local_fragment_1: T.TensorType((16, 128), "float32", "local.fragment") = T.tile_full("float32", T.float32(0.0), 16, 128)
                                T_softmax_expsum_rf_local_fragment_1: T.TensorType((16,), "float32", "local.fragment") = T.tile_full("float32", T.float32(0.0), 16)
                                q_pad_shared_dyn_1: T.TensorType((16, 128), "float16", "shared.dyn") = T.if_then_else(T.tile_unsqueeze(T.tile_arange("int32", 0, 16), T.bool(False), T.bool(True)) < 8, q[0, b1, 0:16, 0:128], T.float16(0.0))
                                @T.for_with_params_body()
                                def loop_fn_j_0_1(j_0_1: T.int32, T_softmax_maxelem_rf_next_local_fragment_1: T.TensorType((16,), "float32", "local.fragment"), T_batch_matmul_NN_rf_local_fragment_2: T.TensorType((16, 128), "float32", "local.fragment"), T_softmax_expsum_rf_local_fragment_2: T.TensorType((16,), "float32", "local.fragment"), T_softmax_maxelem_rf_prev_local_fragment_2: T.TensorType((16,), "float32", "local.fragment")):
                                    with T.block(""):
                                        T.reads(k[0:1, b1:b1 + 1, j_0_0 * 512 + j_0_1 * 128:j_0_0 * 512 + j_0_1 * 128 + 128, 0:128], T_softmax_maxelem_rf_next_local_fragment[0:16], v[0:1, b1:b1 + 1, j_0_0 * 512 + j_0_1 * 128:j_0_0 * 512 + j_0_1 * 128 + 128, 0:128])
                                        T.writes(T_softmax_maxelem_rf_next_local_fragment[0:16])
                                        T_batch_matmul_NT_local_fragment_1: T.TensorType((16, 128), "float32", "local.fragment") = T.tile_full("float32", T.float32(0.0), 16, 128)
                                        @T.for_with_params_body()
                                        def loop_fn_k_0(k_0: T.int32, T_batch_matmul_NT_local_fragment_2: T.TensorType((16, 128), "float32", "local.fragment")):
                                            with T.block(""):
                                                T.reads(k[0:1, b1:b1 + 1, j_0_0 * 512 + j_0_1 * 128:j_0_0 * 512 + j_0_1 * 128 + 128, 0:128])
                                                T.writes()
                                                k_shared_dyn_1: T.TensorType((128, 128), "float16", "shared.dyn") = k[0, b1, j_0_0 * 512 + j_0_1 * 128:j_0_0 * 512 + j_0_1 * 128 + 128, 0:128]
                                                T_batch_matmul_NT_local_fragment_3: T.TensorType((16, 128), "float32", "local.fragment") = T.tile_dot("float32", q_pad_shared_dyn_1, T.tile_permute(k_shared_dyn_1, 1, 0), T_batch_matmul_NT_local_fragment_2)
                                                T.for_yield([T_batch_matmul_NT_local_fragment_3])

                                        T_batch_matmul_NT_local_fragment_4 = T.for_with_params(loop_fn_k_0, min=0, max=1, inits=(T_batch_matmul_NT_local_fragment_1,), annotations={"nautilus_outer_tiled_loop": T.bool(True)})
                                        T_score_mod_local_fragment_1: T.TensorType((16, 128), "float32", "local.fragment") = T.if_then_else(j_0_0 * 512 + j_0_1 * 128 + T.tile_unsqueeze(T.tile_arange("int32", 0, 128), T.bool(True), T.bool(False)) <= T.tile_unsqueeze(T.tile_arange("int32", 0, 16), T.bool(False), T.bool(True)) + 16376, T_batch_matmul_NT_local_fragment_4 * T.float32(0.12751743082459868), T.float32(-340282346638528859811704183484516925440.0))
                                        T_softmax_maxelem_rf_rf_local_fragment_1: T.TensorType((16,), "float32", "local.fragment") = T.tile_reduce(T_score_mod_local_fragment_1, 1, "max")
                                        T_softmax_maxelem_rf_next_local_fragment_2: T.TensorType((16,), "float32", "local.fragment") = T.max(T_softmax_maxelem_rf_prev_local_fragment_2, T_softmax_maxelem_rf_rf_local_fragment_1)
                                        T_softmax_exp_local_fragment_1: T.TensorType((16, 128), "float32", "local.fragment") = T.exp2(T_score_mod_local_fragment_1 - T.tile_unsqueeze(T_softmax_maxelem_rf_next_local_fragment_2, T.bool(False), T.bool(True)))
                                        T_softmax_expsum_rf_rf_local_fragment_1: T.TensorType((16,), "float32", "local.fragment") = T.tile_reduce(T_softmax_exp_local_fragment_1, 1, "sum")
                                        T_softmax_exp_cast_shared_dyn_1: T.TensorType((16, 128), "float16", "shared.dyn") = T.Cast("float16", T_softmax_exp_local_fragment_1)
                                        v_shared_dyn_1: T.TensorType((128, 128), "float16", "shared.dyn") = v[0, b1, j_0_0 * 512 + j_0_1 * 128:j_0_0 * 512 + j_0_1 * 128 + 128, 0:128]
                                        T_batch_matmul_NN_rf_local_fragment_3: T.TensorType((16, 128), "float32", "local.fragment") = T.tile_dot("float32", T_softmax_exp_cast_shared_dyn_1, v_shared_dyn_1, T_batch_matmul_NN_rf_local_fragment_2 * T.tile_unsqueeze(T.exp2(T_softmax_maxelem_rf_prev_local_fragment_2 - T_softmax_maxelem_rf_next_local_fragment_2), T.bool(False), T.bool(True)))
                                        T_softmax_expsum_rf_local_fragment_3: T.TensorType((16,), "float32", "local.fragment") = T_softmax_expsum_rf_local_fragment_2 * T.exp2(T_softmax_maxelem_rf_prev_local_fragment_2 - T_softmax_maxelem_rf_next_local_fragment_2) + T_softmax_expsum_rf_rf_local_fragment_1
                                        T_softmax_maxelem_rf_prev_local_fragment_3: T.TensorType((16,), "float32", "local.fragment") = T_softmax_maxelem_rf_next_local_fragment_2
                                        T.for_yield([T_softmax_maxelem_rf_next_local_fragment_2, T_batch_matmul_NN_rf_local_fragment_3, T_softmax_expsum_rf_local_fragment_3, T_softmax_maxelem_rf_prev_local_fragment_3])

                                T_softmax_maxelem_rf_next_local_fragment_3, T_batch_matmul_NN_rf_local_fragment_4, T_softmax_expsum_rf_local_fragment_4, T_softmax_maxelem_rf_prev_local_fragment_4 = T.for_with_params(loop_fn_j_0_1, min=0, max=4, inits=(T_softmax_maxelem_rf_next_local_fragment[0:16], T_batch_matmul_NN_rf_local_fragment_1, T_softmax_expsum_rf_local_fragment_1, T_softmax_maxelem_rf_prev_local_fragment_1), annotations={"nautilus_block_order": 1, "nautilus_fusion_role": "rolling", "nautilus_outer_tiled_loop": T.bool(True), "tir.loop_original_bounds": T.Range(0, 4)})
                                T_batch_matmul_NN_rf[j_0_0, b1, 0:16, 0:128] = T_batch_matmul_NN_rf_local_fragment_4
                                T_softmax_expsum_rf[j_0_0, b1, 0:16] = T_softmax_expsum_rf_local_fragment_4
                                T_softmax_maxelem_rf_next_global[j_0_0, b1, 0:16] = T_softmax_maxelem_rf_next_local_fragment_3
            for ax1_ax2_ax3_fused in T.thread_binding(64, thread="blockIdx.x", annotations={"nautilus_block_order": 0, "nautilus_split_k_second": T.bool(True), "pragma_nautilus_fold_accumulators": 1, "pragma_num_warps": 2, "pragma_tile_optimizer": "tilelang", "pragma_tl_num_stages": 1, "pragma_tl_tma": 0, "pragma_tl_warp_specialize": 0, "pragma_trcg_pointer_method": 3, "pragma_triton_enable_warp_specialization": T.bool(True), "pragma_triton_force_membar": T.bool(False), "pragma_triton_math_wg_pipe": T.bool(True), "pragma_triton_mma_depth": 1, "pragma_triton_num_stages": 2, "pragma_triton_num_warps": 2, "pragma_triton_use_ttg_ws": T.bool(False)}):
                with T.block(""):
                    T.reads(T_softmax_maxelem_rf_next_global[0:32, ax1_ax2_ax3_fused // 16:ax1_ax2_ax3_fused // 16 + 1, ax1_ax2_ax3_fused % 16:ax1_ax2_ax3_fused % 16 + 1], T_batch_matmul_NN_rf[0:32, ax1_ax2_ax3_fused // 16:ax1_ax2_ax3_fused // 16 + 1, ax1_ax2_ax3_fused % 16:ax1_ax2_ax3_fused % 16 + 1, 0:128], T_softmax_expsum_rf[0:32, ax1_ax2_ax3_fused // 16:ax1_ax2_ax3_fused // 16 + 1, ax1_ax2_ax3_fused % 16:ax1_ax2_ax3_fused % 16 + 1])
                    T.writes(T_cast[0:1, ax1_ax2_ax3_fused // 16:ax1_ax2_ax3_fused // 16 + 1, ax1_ax2_ax3_fused % 16:ax1_ax2_ax3_fused % 16 + 1, 0:128])
                    T_softmax_maxelem_rf_next_global_local_fragment_1: T.TensorType((32,), "float32", "local.fragment") = T_softmax_maxelem_rf_next_global[0:32, ax1_ax2_ax3_fused // 16, ax1_ax2_ax3_fused % 16]
                    T_softmax_maxelem_local_fragment_1: T.TensorType((), "float32", "local.fragment") = T.tile_reduce(T_softmax_maxelem_rf_next_global_local_fragment_1, 0, "max")
                    T_batch_matmul_NN_rf_local_fragment_5: T.TensorType((32, 128), "float32", "local.fragment") = T_batch_matmul_NN_rf[0:32, ax1_ax2_ax3_fused // 16, ax1_ax2_ax3_fused % 16, 0:128]
                    T_batch_matmul_NN_local_fragment_1: T.TensorType((128,), "float32", "local.fragment") = T.tile_reduce(T_batch_matmul_NN_rf_local_fragment_5 * T.exp2(T.tile_unsqueeze(T_softmax_maxelem_rf_next_global_local_fragment_1, T.bool(False), T.bool(True)) - T.tile_unsqueeze(T_softmax_maxelem_local_fragment_1, T.bool(True), T.bool(True))), 0, "sum")
                    T_softmax_expsum_rf_local_fragment_5: T.TensorType((32,), "float32", "local.fragment") = T_softmax_expsum_rf[0:32, ax1_ax2_ax3_fused // 16, ax1_ax2_ax3_fused % 16]
                    T_softmax_expsum_local_fragment_1: T.TensorType((), "float32", "local.fragment") = T.tile_reduce(T_softmax_expsum_rf_local_fragment_5 * T.exp2(T_softmax_maxelem_rf_next_global_local_fragment_1 - T.tile_unsqueeze(T_softmax_maxelem_local_fragment_1, T.bool(True))), 0, "sum")
                    T_softmax_norm_local_fragment_1: T.TensorType((128,), "float32", "local.fragment") = T_batch_matmul_NN_local_fragment_1 / T.tile_unsqueeze(T_softmax_expsum_local_fragment_1, T.bool(True))
                    T_cast_pad_local_fragment_1: T.TensorType((128,), "float16", "local.fragment") = T.Cast("float16", T_softmax_norm_local_fragment_1)
                    T_cast_shared_dyn_1: T.TensorType((128,), "float16", "shared.dyn") = T_cast_pad_local_fragment_1
                    if ax1_ax2_ax3_fused % 16 < 8:
                        T_cast[0, ax1_ax2_ax3_fused // 16, ax1_ax2_ax3_fused % 16, 0:128] = T_cast_shared_dyn_1
                        T_cast_1: T.TensorType((128,), "float16", "global") = T_cast_shared_dyn_1
                        T.evaluate(0)