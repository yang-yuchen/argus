# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def neptune_attention(q: T.Buffer((1, 32, 1, 128), "float8_e4m3fn"), k: T.Buffer((1, 32, 8192, 128), "float8_e4m3fn"), v: T.Buffer((1, 32, 128, 8192), "float8_e4m3fn"), T_cast: T.Buffer((1, 32, 1, 128), "float16")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True, "tir.tile_expr_form": True})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"nautilus_applied_rules": "tile(T_batch_matmul_NT, SRSR); RCA(T_score_mod, j_0); RU(T_softmax_maxelem, j_0); RCA(T_softmax_exp, ax1_ax2_ax3_fused); RCA(T_softmax_exp_cast, ax1_ax2_ax3_fused); RU(T_batch_matmul_NT_1, j_0); RU(T_softmax_expsum, j_0); RCA(T_softmax_norm, ax1_ax2_ax3_fused); RCA(T_cast, ax1_ax2_ax3_fused); ", "schedule_nums": "0,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,272,273,275,276,277,279,281,282,283,284,713,714,715,716,717,718,719,720,721,722,1523,1524,1525,1526,1527,1528,1529,1530,1531,1532,2093,2094,2095,2096,2097,2098,2099,2100,2101,2102,2903,2904,2905,2906,2907,2908,2909,2910,2911,2912,"})
            T_softmax_maxelem_rf = T.alloc_buffer((64, 32))
            T_batch_matmul_NT_rf = T.alloc_buffer((64, 32, 128))
            T_softmax_expsum_rf = T.alloc_buffer((64, 32))
            for b0 in T.thread_binding(1, thread="blockIdx.z", annotations={"pragma_nautilus_fold_accumulators": 0, "pragma_num_warps": 4, "pragma_tile_optimizer": "tilelang", "pragma_tl_num_stages": 0, "pragma_tl_tma": 0, "pragma_tl_warp_specialize": 1, "pragma_trcg_pointer_method": 3, "pragma_triton_enable_warp_specialization": T.bool(True), "pragma_triton_force_membar": T.bool(False), "pragma_triton_math_wg_pipe": T.bool(True), "pragma_triton_mma_depth": 1, "pragma_triton_num_stages": 2, "pragma_triton_num_warps": 4, "pragma_triton_use_ttg_ws": T.bool(False)}):
                for b1 in T.thread_binding(32, thread="blockIdx.y"):
                    for i in range(1):
                        for j_0 in T.thread_binding(64, thread="blockIdx.x", annotations={"nautilus_block_order": 1, "nautilus_fusion_role": "split", "nautilus_outer_tiled_loop": T.bool(True)}):
                            with T.block(""):
                                T.reads(q[0:1, b1:b1 + 1, 0:1, 0:128], k[0:1, b1:b1 + 1, j_0 * 128:j_0 * 128 + 128, 0:128], v[0:1, b1:b1 + 1, 0:128, j_0 * 128:j_0 * 128 + 128])
                                T.writes(T_softmax_expsum_rf[j_0:j_0 + 1, b1:b1 + 1], T_batch_matmul_NT_rf[j_0:j_0 + 1, b1:b1 + 1, 0:128], T_softmax_maxelem_rf[j_0:j_0 + 1, b1:b1 + 1])
                                T_batch_matmul_NT_local_fragment_1: T.TensorType((128,), "float32", "local.fragment") = T.tile_full("float32", T.float32(0.0), 128)
                                @T.for_with_params_body()
                                def loop_fn_k_0(k_0: T.int32, T_batch_matmul_NT_local_fragment_2: T.TensorType((128,), "float32", "local.fragment")):
                                    with T.block(""):
                                        T.reads(q[0:1, b1:b1 + 1, 0:1, 0:128], k[0:1, b1:b1 + 1, j_0 * 128:j_0 * 128 + 128, 0:128])
                                        T.writes()
                                        q_shared_dyn_1: T.TensorType((128,), "float8_e4m3fn", "shared.dyn") = q[0, b1, 0, 0:128]
                                        k_shared_dyn_1: T.TensorType((128, 128), "float8_e4m3fn", "shared.dyn") = k[0, b1, j_0 * 128:j_0 * 128 + 128, 0:128]
                                        T_batch_matmul_NT_local_fragment_3: T.TensorType((128,), "float32", "local.fragment") = T_batch_matmul_NT_local_fragment_2 + T.tile_reduce(T.Cast("float32", q_shared_dyn_1) * T.Cast("float32", k_shared_dyn_1), 1, "sum")
                                        T.for_yield([T_batch_matmul_NT_local_fragment_3])

                                T_batch_matmul_NT_local_fragment_4 = T.for_with_params(loop_fn_k_0, min=0, max=1, inits=(T_batch_matmul_NT_local_fragment_1,), annotations={"nautilus_fusion_role": "split", "nautilus_outer_tiled_loop": T.bool(True)})
                                T_score_mod_local_fragment_1: T.TensorType((128,), "float32", "local.fragment") = T_batch_matmul_NT_local_fragment_4 * T.float32(0.12751743082459868)
                                T_softmax_maxelem_rf_shared_1: T.TensorType((), "float32", "shared") = T.tile_reduce(T_score_mod_local_fragment_1, 0, "max")
                                T_softmax_exp_local_fragment_1: T.TensorType((128,), "float32", "local.fragment") = T.exp2(T_score_mod_local_fragment_1 - T.tile_unsqueeze(T_softmax_maxelem_rf_shared_1, T.bool(True)))
                                T_softmax_expsum_rf[j_0, b1] = T.tile_reduce(T_softmax_exp_local_fragment_1, 0, "sum")
                                T_softmax_exp_cast_shared_dyn_1: T.TensorType((128,), "float8_e4m3fn", "shared.dyn") = T.Cast("float8_e4m3fn", T_softmax_exp_local_fragment_1)
                                T_batch_matmul_NT_rf[j_0, b1, 0:128] = T.tile_reduce(T.Cast("float32", T_softmax_exp_cast_shared_dyn_1) * T.Cast("float32", v[0, b1, 0:128, j_0 * 128:j_0 * 128 + 128]), 1, "sum")
                                T_softmax_maxelem_rf[j_0, b1] = T_softmax_maxelem_rf_shared_1
            for ax1_ax2_ax3_fused in T.thread_binding(32, thread="blockIdx.x", annotations={"nautilus_block_order": 0, "nautilus_split_k_second": T.bool(True), "pragma_nautilus_fold_accumulators": 1, "pragma_num_warps": 4, "pragma_tile_optimizer": "tilelang", "pragma_tl_num_stages": 0, "pragma_tl_tma": 0, "pragma_tl_warp_specialize": 1, "pragma_trcg_pointer_method": 3, "pragma_triton_enable_warp_specialization": T.bool(True), "pragma_triton_force_membar": T.bool(False), "pragma_triton_math_wg_pipe": T.bool(True), "pragma_triton_mma_depth": 1, "pragma_triton_num_stages": 2, "pragma_triton_num_warps": 4, "pragma_triton_use_ttg_ws": T.bool(False)}):
                with T.block(""):
                    T.reads(T_softmax_maxelem_rf[0:64, ax1_ax2_ax3_fused:ax1_ax2_ax3_fused + 1], T_batch_matmul_NT_rf[0:64, ax1_ax2_ax3_fused:ax1_ax2_ax3_fused + 1, 0:128], T_softmax_expsum_rf[0:64, ax1_ax2_ax3_fused:ax1_ax2_ax3_fused + 1])
                    T.writes(T_cast[0:1, ax1_ax2_ax3_fused:ax1_ax2_ax3_fused + 1, 0:1, 0:128])
                    T_softmax_maxelem_rf_local_fragment_1: T.TensorType((64,), "float32", "local.fragment") = T_softmax_maxelem_rf[0:64, ax1_ax2_ax3_fused]
                    T_softmax_maxelem_local_fragment_1: T.TensorType((), "float32", "local.fragment") = T.tile_reduce(T_softmax_maxelem_rf_local_fragment_1, 0, "max")
                    T_batch_matmul_NT_rf_local_fragment_1: T.TensorType((64, 128), "float32", "local.fragment") = T_batch_matmul_NT_rf[0:64, ax1_ax2_ax3_fused, 0:128]
                    T_batch_matmul_NT_local_fragment_5: T.TensorType((128,), "float32", "local.fragment") = T.tile_reduce(T_batch_matmul_NT_rf_local_fragment_1 * T.exp2(T.tile_unsqueeze(T_softmax_maxelem_rf_local_fragment_1, T.bool(False), T.bool(True)) - T.tile_unsqueeze(T_softmax_maxelem_local_fragment_1, T.bool(True), T.bool(True))), 0, "sum")
                    T_softmax_expsum_rf_local_fragment_1: T.TensorType((64,), "float32", "local.fragment") = T_softmax_expsum_rf[0:64, ax1_ax2_ax3_fused]
                    T_softmax_expsum_local_fragment_1: T.TensorType((), "float32", "local.fragment") = T.tile_reduce(T_softmax_expsum_rf_local_fragment_1 * T.exp2(T_softmax_maxelem_rf_local_fragment_1 - T.tile_unsqueeze(T_softmax_maxelem_local_fragment_1, T.bool(True))), 0, "sum")
                    T_softmax_norm_local_fragment_1: T.TensorType((128,), "float32", "local.fragment") = T_batch_matmul_NT_local_fragment_5 / T.tile_unsqueeze(T_softmax_expsum_local_fragment_1, T.bool(True))
                    T_cast_shared_dyn_1: T.TensorType((128,), "float16", "shared.dyn") = T.Cast("float16", T_softmax_norm_local_fragment_1)
                    T_cast[0, ax1_ax2_ax3_fused, 0, 0:128] = T_cast_shared_dyn_1
                    T_cast_1: T.TensorType((128,), "float16", "global") = T_cast_shared_dyn_1
                    T.evaluate(0)