# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def neptune_attention(q: T.Buffer((1, 4, 64, 128), "float16"), k: T.Buffer((1, 4, 256, 128), "float16"), v: T.Buffer((1, 4, 256, 128), "float16"), T_cast: T.Buffer((1, 4, 64, 128), "float16")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True, "tir.tile_expr_form": True})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"nautilus_applied_rules": "tile(T_batch_matmul_NT, SRSR); RCA(T_score_mod, j_0); RU(T_softmax_maxelem, j_0); RCA(T_softmax_exp, ax1_ax2_ax3_fused); RCA(T_softmax_exp_cast, ax1_ax2_ax3_fused); RU(T_batch_matmul_NN, j_0); RU(T_softmax_expsum, j_0); RCA(T_softmax_norm, ax1_ax2_ax3_fused); RCA(T_cast, ax1_ax2_ax3_fused); ", "schedule_nums": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,25,26,28,29,30,32,34,35,36,37,570,571,572,573,574,575,576,577,578,579,580,581,582,583,584,585,586,587,588,589,1290,1291,1292,1293,1294,1295,1296,1297,1298,1299,1300,1301,1302,1303,1304,1305,1306,1307,1308,1309,"})
            T_softmax_maxelem_rf = T.alloc_buffer((8, 4, 64))
            T_batch_matmul_NN_rf = T.alloc_buffer((8, 4, 64, 128))
            T_softmax_expsum_rf = T.alloc_buffer((8, 4, 64))
            for b0_b1_fused in T.thread_binding(4, thread="blockIdx.z", annotations={"pragma_nautilus_fold_accumulators": 0, "pragma_num_warps": 2, "pragma_tile_optimizer": "tilelang", "pragma_tl_num_stages": 0, "pragma_tl_tma": 0, "pragma_tl_warp_specialize": 1, "pragma_trcg_pointer_method": 3, "pragma_triton_enable_warp_specialization": T.bool(True), "pragma_triton_force_membar": T.bool(False), "pragma_triton_math_wg_pipe": T.bool(True), "pragma_triton_mma_depth": 1, "pragma_triton_num_stages": 2, "pragma_triton_num_warps": 2, "pragma_triton_use_ttg_ws": T.bool(False)}):
                for i in T.thread_binding(64, thread="blockIdx.y"):
                    for j_0 in T.thread_binding(8, thread="blockIdx.x", annotations={"nautilus_block_order": 1, "nautilus_fusion_role": "split", "nautilus_outer_tiled_loop": T.bool(True)}):
                        with T.block(""):
                            T.reads(q[0:1, b0_b1_fused:b0_b1_fused + 1, i:i + 1, 0:128], k[0:1, b0_b1_fused:b0_b1_fused + 1, j_0 * 32:j_0 * 32 + 32, 0:128], v[0:1, b0_b1_fused:b0_b1_fused + 1, j_0 * 32:j_0 * 32 + 32, 0:128])
                            T.writes(T_softmax_expsum_rf[j_0:j_0 + 1, b0_b1_fused:b0_b1_fused + 1, i:i + 1], T_batch_matmul_NN_rf[j_0:j_0 + 1, b0_b1_fused:b0_b1_fused + 1, i:i + 1, 0:128], T_softmax_maxelem_rf[j_0:j_0 + 1, b0_b1_fused:b0_b1_fused + 1, i:i + 1])
                            T_batch_matmul_NT_local_fragment_1: T.TensorType((32,), "float32", "local.fragment") = T.tile_full("float32", T.float32(0.0), 32)
                            @T.for_with_params_body()
                            def loop_fn_k_0(k_0: T.int32, T_batch_matmul_NT_local_fragment_2: T.TensorType((32,), "float32", "local.fragment")):
                                with T.block(""):
                                    T.reads(q[0:1, b0_b1_fused:b0_b1_fused + 1, i:i + 1, k_0 * 32:k_0 * 32 + 32], k[0:1, b0_b1_fused:b0_b1_fused + 1, j_0 * 32:j_0 * 32 + 32, k_0 * 32:k_0 * 32 + 32])
                                    T.writes()
                                    q_shared_dyn_1: T.TensorType((32,), "float16", "shared.dyn") = q[0, b0_b1_fused, i, k_0 * 32:k_0 * 32 + 32]
                                    k_shared_dyn_1: T.TensorType((32, 32), "float16", "shared.dyn") = k[0, b0_b1_fused, j_0 * 32:j_0 * 32 + 32, k_0 * 32:k_0 * 32 + 32]
                                    T_batch_matmul_NT_local_fragment_3: T.TensorType((32,), "float32", "local.fragment") = T_batch_matmul_NT_local_fragment_2 + T.tile_reduce(T.Cast("float32", q_shared_dyn_1) * T.Cast("float32", k_shared_dyn_1), 1, "sum")
                                    T.for_yield([T_batch_matmul_NT_local_fragment_3])

                            T_batch_matmul_NT_local_fragment_4 = T.for_with_params(loop_fn_k_0, min=0, max=4, inits=(T_batch_matmul_NT_local_fragment_1,), annotations={"nautilus_fusion_role": "split", "nautilus_outer_tiled_loop": T.bool(True)})
                            T_score_mod_local_fragment_1: T.TensorType((32,), "float32", "local.fragment") = T.if_then_else(j_0 * 32 + T.tile_arange("int32", 0, 32) <= i + 192, T_batch_matmul_NT_local_fragment_4 * T.float32(0.12751743082459868), T.float32(-340282346638528859811704183484516925440.0))
                            T_softmax_maxelem_rf_shared_1: T.TensorType((), "float32", "shared") = T.tile_reduce(T_score_mod_local_fragment_1, 0, "max")
                            T_softmax_exp_local_fragment_1: T.TensorType((32,), "float32", "local.fragment") = T.exp2(T_score_mod_local_fragment_1 - T.tile_unsqueeze(T_softmax_maxelem_rf_shared_1, T.bool(True)))
                            T_softmax_expsum_rf[j_0, b0_b1_fused, i] = T.tile_reduce(T_softmax_exp_local_fragment_1, 0, "sum")
                            T_softmax_exp_cast_shared_dyn_1: T.TensorType((32,), "float16", "shared.dyn") = T.Cast("float16", T_softmax_exp_local_fragment_1)
                            T_batch_matmul_NN_rf[j_0, b0_b1_fused, i, 0:128] = T.tile_reduce(T.Cast("float32", T_softmax_exp_cast_shared_dyn_1) * T.Cast("float32", T.tile_permute(v[0, b0_b1_fused, j_0 * 32:j_0 * 32 + 32, 0:128], 1, 0)), 1, "sum")
                            T_softmax_maxelem_rf[j_0, b0_b1_fused, i] = T_softmax_maxelem_rf_shared_1
            for ax1_ax2_ax3_fused in T.thread_binding(256, thread="blockIdx.x", annotations={"nautilus_block_order": 1, "nautilus_split_k_second": T.bool(True), "pragma_nautilus_fold_accumulators": 1, "pragma_num_warps": 2, "pragma_tile_optimizer": "tilelang", "pragma_tl_num_stages": 2, "pragma_tl_tma": 0, "pragma_tl_warp_specialize": 1, "pragma_trcg_pointer_method": 3, "pragma_triton_enable_warp_specialization": T.bool(True), "pragma_triton_force_membar": T.bool(False), "pragma_triton_math_wg_pipe": T.bool(True), "pragma_triton_mma_depth": 1, "pragma_triton_num_stages": 2, "pragma_triton_num_warps": 2, "pragma_triton_use_ttg_ws": T.bool(False)}):
                with T.block(""):
                    T.reads(T_softmax_maxelem_rf[0:8, ax1_ax2_ax3_fused // 64:ax1_ax2_ax3_fused // 64 + 1, ax1_ax2_ax3_fused % 64:ax1_ax2_ax3_fused % 64 + 1], T_softmax_expsum_rf[0:8, ax1_ax2_ax3_fused // 64:ax1_ax2_ax3_fused // 64 + 1, ax1_ax2_ax3_fused % 64:ax1_ax2_ax3_fused % 64 + 1], T_batch_matmul_NN_rf[0:8, ax1_ax2_ax3_fused // 64:ax1_ax2_ax3_fused // 64 + 1, ax1_ax2_ax3_fused % 64:ax1_ax2_ax3_fused % 64 + 1, 0:128])
                    T.writes(T_cast[0:1, ax1_ax2_ax3_fused // 64:ax1_ax2_ax3_fused // 64 + 1, ax1_ax2_ax3_fused % 64:ax1_ax2_ax3_fused % 64 + 1, 0:128])
                    T_softmax_maxelem_rf_local_fragment_1: T.TensorType((8,), "float32", "local.fragment") = T_softmax_maxelem_rf[0:8, ax1_ax2_ax3_fused // 64, ax1_ax2_ax3_fused % 64]
                    T_softmax_maxelem_local_fragment_1: T.TensorType((), "float32", "local.fragment") = T.tile_reduce(T_softmax_maxelem_rf_local_fragment_1, 0, "max")
                    T_softmax_expsum_rf_local_fragment_1: T.TensorType((8,), "float32", "local.fragment") = T_softmax_expsum_rf[0:8, ax1_ax2_ax3_fused // 64, ax1_ax2_ax3_fused % 64]
                    T_softmax_expsum_local_fragment_1: T.TensorType((), "float32", "local.fragment") = T.tile_reduce(T_softmax_expsum_rf_local_fragment_1 * T.exp2(T_softmax_maxelem_rf_local_fragment_1 - T.tile_unsqueeze(T_softmax_maxelem_local_fragment_1, T.bool(True))), 0, "sum")
                    T_batch_matmul_NN_rf_local_fragment_1: T.TensorType((8, 128), "float32", "local.fragment") = T_batch_matmul_NN_rf[0:8, ax1_ax2_ax3_fused // 64, ax1_ax2_ax3_fused % 64, 0:128]
                    T_batch_matmul_NN_local_fragment_1: T.TensorType((128,), "float32", "local.fragment") = T.tile_reduce(T_batch_matmul_NN_rf_local_fragment_1 * T.exp2(T.tile_unsqueeze(T_softmax_maxelem_rf_local_fragment_1, T.bool(False), T.bool(True)) - T.tile_unsqueeze(T_softmax_maxelem_local_fragment_1, T.bool(True), T.bool(True))), 0, "sum")
                    T_softmax_norm_local_fragment_1: T.TensorType((128,), "float32", "local.fragment") = T_batch_matmul_NN_local_fragment_1 / T.tile_unsqueeze(T_softmax_expsum_local_fragment_1, T.bool(True))
                    T_cast_shared_dyn_1: T.TensorType((128,), "float16", "shared.dyn") = T.Cast("float16", T_softmax_norm_local_fragment_1)
                    T_cast[0, ax1_ax2_ax3_fused // 64, ax1_ax2_ax3_fused % 64, 0:128] = T_cast_shared_dyn_1
                    T_cast_1: T.TensorType((128,), "float16", "global") = T_cast_shared_dyn_1
                    T.evaluate(0)