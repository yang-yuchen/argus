#include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel_kernel(float* __restrict__ T_batch_matmul_NN_rf, float* __restrict__ T_softmax_expsum_rf, float* __restrict__ T_softmax_maxelem_rf_next_global, __grid_constant__ const CUtensorMap k_desc, half_t* __restrict__ q, __grid_constant__ const CUtensorMap v_desc);
extern "C" __global__ void main_kernel_1_kernel(float* __restrict__ T_batch_matmul_NN_rf, half_t* __restrict__ T_cast, float* __restrict__ T_softmax_expsum_rf, float* __restrict__ T_softmax_maxelem_rf_next_global);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel_kernel(float* __restrict__ T_batch_matmul_NN_rf, float* __restrict__ T_softmax_expsum_rf, float* __restrict__ T_softmax_maxelem_rf_next_global, __grid_constant__ const CUtensorMap k_desc, half_t* __restrict__ q, __grid_constant__ const CUtensorMap v_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float T_softmax_maxelem_rf_prev_local_fragment_1[2];
  float T_batch_matmul_NN_rf_local_fragment_1[32];
  float T_softmax_expsum_rf_local_fragment_1[2];
  float T_softmax_maxelem_rf_next_local_fragment[2];
  float T_softmax_maxelem_rf_next_local_fragment_1[2];
  float T_batch_matmul_NT_local_fragment_1[8];
  float T_score_mod_local_fragment_1[8];
  float T_softmax_maxelem_rf_rf_local_fragment_1[2];
  float T_softmax_exp_local_fragment_1[8];
  float T_softmax_expsum_rf_rf_local_fragment_1[2];
  __shared__ uint64_t mbarrier_mem[5];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(k_desc);
    tl::prefetch_tma_descriptor(v_desc);
    mbarrier[0].init(64);
    mbarrier[1].init(64);
    mbarrier[2].init(64);
    mbarrier[3].init(64);
    mbarrier[4].init(64);
  }
  __syncthreads();
  if (64 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint4 condval;
      if ((i < 1)) {
        condval = *(uint4*)(q + ((((((int)blockIdx.y) * 512) + (i * 512)) + (((int)threadIdx.x) * 8)) - 512));
      } else {
        condval = make_uint4(__pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)));
      }
      *(uint4*)(((half_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 1024) + (i * 256)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 7936)) = condval;
    }
    tl::fence_proxy_async();
    tl::mbarrier_cp_async_arrive(mbarrier[4]);
    mbarrier[4].arrive();
    for (int j_0_1 = 0; j_0_1 < 2; ++j_0_1) {
      mbarrier[2].wait((j_0_1 ^ 1));
      if (tl::tl_shuffle_elect<64>()) {
        mbarrier[0].expect_transaction(8192);
        tl::tma_load(k_desc, mbarrier[0], (&(((half_t*)buf_dyn_shmem)[0])), 0, ((((int)blockIdx.x) * 64) + (j_0_1 * 32)), ((int)blockIdx.y), 0);
        tl::tma_load(k_desc, mbarrier[0], (&(((half_t*)buf_dyn_shmem)[2048])), 64, ((((int)blockIdx.x) * 64) + (j_0_1 * 32)), ((int)blockIdx.y), 0);
      }
      mbarrier[0].arrive();
      mbarrier[3].wait((j_0_1 ^ 1));
      if (tl::tl_shuffle_elect<64>()) {
        mbarrier[1].expect_transaction(8192);
        tl::tma_load(v_desc, mbarrier[1], (&(((half_t*)buf_dyn_shmem)[4096])), 0, ((((int)blockIdx.x) * 64) + (j_0_1 * 32)), ((int)blockIdx.y), 0);
        tl::tma_load(v_desc, mbarrier[1], (&(((half_t*)buf_dyn_shmem)[6144])), 64, ((((int)blockIdx.x) * 64) + (j_0_1 * 32)), ((int)blockIdx.y), 0);
      }
      mbarrier[1].arrive();
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      T_softmax_maxelem_rf_prev_local_fragment_1[i_1] = -0x1.fffffep+127f/*-3.402823e+38*/;
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      *(float2*)(T_batch_matmul_NN_rf_local_fragment_1 + (i_2 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      T_softmax_expsum_rf_local_fragment_1[i_3] = 0x0p+0f/*0.000000e+00*/;
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      T_softmax_maxelem_rf_next_local_fragment_1[i_4] = T_softmax_maxelem_rf_next_local_fragment[i_4];
    }
    tl::fence_proxy_async();
    mbarrier[4].wait(0);
    for (int j_0_1_1 = 0; j_0_1_1 < 2; ++j_0_1_1) {
      #pragma unroll
      for (int i_5 = 0; i_5 < 4; ++i_5) {
        *(float2*)(T_batch_matmul_NT_local_fragment_1 + (i_5 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
      }
      tl::fence_proxy_async();
      mbarrier[0].wait(j_0_1_1);
      tl::__sync_thread_partial<3, 64>();
      tl::gemm_ss<16, 32, 128, 1, 2, 0, 1, 0, 128, 128, 0, 0, false>((&(((half_t*)buf_dyn_shmem)[8192])), (&(((half_t*)buf_dyn_shmem)[0])), (&(T_batch_matmul_NT_local_fragment_1[0])));
      mbarrier[2].arrive();
      #pragma unroll
      for (int i_6 = 0; i_6 < 8; ++i_6) {
        float condval_1;
        if ((((((((((int)blockIdx.x) * 64) + (j_0_1_1 * 32)) + ((i_6 >> 2) * 16)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1)) <= (((((i_6 & 3) >> 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 4092))) {
          condval_1 = (T_batch_matmul_NT_local_fragment_1[i_6] * 0x1.0527dbd87e24dp-3f/*1.275174e-01*/);
        } else {
          condval_1 = -0x1.fffffep+127f/*-3.402823e+38*/;
        }
        T_score_mod_local_fragment_1[i_6] = condval_1;
      }
      tl::__sync_thread_partial<3, 64>();
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        T_softmax_maxelem_rf_rf_local_fragment_1[i_7] = -CUDART_INF_F;
        #pragma unroll
        for (int rv = 0; rv < 4; ++rv) {
          T_softmax_maxelem_rf_rf_local_fragment_1[i_7] = max(T_softmax_maxelem_rf_rf_local_fragment_1[i_7], T_score_mod_local_fragment_1[((((rv & 1) * 4) + (i_7 * 2)) + (rv >> 1))]);
        }
        T_softmax_maxelem_rf_rf_local_fragment_1[i_7] = tl::AllReduce<tl::MaxOp, 64, 32, 0, 64>::run_hopper(T_softmax_maxelem_rf_rf_local_fragment_1[i_7], (&(((float*)buf_dyn_shmem)[5376])));
        T_softmax_maxelem_rf_rf_local_fragment_1[i_7] = tl::AllReduce<tl::MaxOp, 4, 1, 0, 64>::run_hopper(T_softmax_maxelem_rf_rf_local_fragment_1[i_7]);
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        T_softmax_maxelem_rf_next_local_fragment_1[i_8] = max(T_softmax_maxelem_rf_prev_local_fragment_1[i_8], T_softmax_maxelem_rf_rf_local_fragment_1[i_8]);
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 8; ++i_9) {
        T_softmax_exp_local_fragment_1[i_9] = exp2f((T_score_mod_local_fragment_1[i_9] - T_softmax_maxelem_rf_next_local_fragment_1[((i_9 & 3) >> 1)]));
      }
      tl::ptx_stmatrix_x4((&(((half_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 10240)])), __pack_half2(((half_t)T_softmax_exp_local_fragment_1[0]), ((half_t)T_softmax_exp_local_fragment_1[1])), __pack_half2(((half_t)T_softmax_exp_local_fragment_1[2]), ((half_t)T_softmax_exp_local_fragment_1[3])), __pack_half2(((half_t)T_softmax_exp_local_fragment_1[4]), ((half_t)T_softmax_exp_local_fragment_1[5])), __pack_half2(((half_t)T_softmax_exp_local_fragment_1[6]), ((half_t)T_softmax_exp_local_fragment_1[7])));
      #pragma unroll
      for (int i_10 = 0; i_10 < 32; ++i_10) {
        T_batch_matmul_NN_rf_local_fragment_1[i_10] = (T_batch_matmul_NN_rf_local_fragment_1[i_10] * exp2f((T_softmax_maxelem_rf_prev_local_fragment_1[((i_10 & 3) >> 1)] - T_softmax_maxelem_rf_next_local_fragment_1[((i_10 & 3) >> 1)])));
      }
      tl::fence_proxy_async();
      mbarrier[1].wait(j_0_1_1);
      tl::__sync_thread_partial<3, 64>();
      tl::gemm_ss<16, 128, 32, 1, 2, 0, 0, 0, 32, 128, 0, 0, false>((&(((half_t*)buf_dyn_shmem)[10240])), (&(((half_t*)buf_dyn_shmem)[4096])), (&(T_batch_matmul_NN_rf_local_fragment_1[0])));
      mbarrier[3].arrive();
      tl::__sync_thread_partial<3, 64>();
      #pragma unroll
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        T_softmax_expsum_rf_rf_local_fragment_1[i_11] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
          T_softmax_expsum_rf_rf_local_fragment_1[i_11] = (T_softmax_expsum_rf_rf_local_fragment_1[i_11] + T_softmax_exp_local_fragment_1[((((rv_1 & 1) * 4) + (i_11 * 2)) + (rv_1 >> 1))]);
        }
        T_softmax_expsum_rf_rf_local_fragment_1[i_11] = tl::AllReduce<tl::SumOp, 64, 32, 0, 64>::run_hopper(T_softmax_expsum_rf_rf_local_fragment_1[i_11], (&(((float*)buf_dyn_shmem)[5440])));
        T_softmax_expsum_rf_rf_local_fragment_1[i_11] = tl::AllReduce<tl::SumOp, 4, 1, 0, 64>::run_hopper(T_softmax_expsum_rf_rf_local_fragment_1[i_11]);
      }
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        T_softmax_expsum_rf_local_fragment_1[i_12] = ((T_softmax_expsum_rf_local_fragment_1[i_12] * exp2f((T_softmax_maxelem_rf_prev_local_fragment_1[i_12] - T_softmax_maxelem_rf_next_local_fragment_1[i_12]))) + T_softmax_expsum_rf_rf_local_fragment_1[i_12]);
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        T_softmax_maxelem_rf_prev_local_fragment_1[i_13] = T_softmax_maxelem_rf_next_local_fragment_1[i_13];
      }
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 16; ++i_14) {
      *(float2*)(T_batch_matmul_NN_rf + (((((((((int)blockIdx.x) * 8192) + (((int)blockIdx.y) * 2048)) + ((i_14 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_14 >> 1) * 16)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(T_batch_matmul_NN_rf_local_fragment_1 + (i_14 * 2));
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      T_softmax_expsum_rf[((((((int)blockIdx.x) * 64) + (((int)blockIdx.y) * 16)) + (i_15 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = T_softmax_expsum_rf_local_fragment_1[i_15];
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 2; ++i_16) {
      T_softmax_maxelem_rf_next_global[((((((int)blockIdx.x) * 64) + (((int)blockIdx.y) * 16)) + (i_16 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = T_softmax_maxelem_rf_next_local_fragment_1[i_16];
    }
  }
}

extern "C" __global__ void __launch_bounds__(128, 1) main_kernel_1_kernel(float* __restrict__ T_batch_matmul_NN_rf, half_t* __restrict__ T_cast, float* __restrict__ T_softmax_expsum_rf, float* __restrict__ T_softmax_maxelem_rf_next_global) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float T_softmax_maxelem_rf_next_global_local_fragment_1[16];
  float T_softmax_maxelem_local_fragment_1[1];
  float T_batch_matmul_NN_rf_local_fragment_5[64];
  float reduce_input[64];
  float T_batch_matmul_NN_local_fragment_1[4];
  float T_softmax_expsum_rf_local_fragment_5[16];
  float reduce_input_1[16];
  float T_softmax_expsum_local_fragment_1[1];
  float T_softmax_norm_local_fragment_1[4];
  half_t T_cast_pad_local_fragment_1[4];
  if ((((int)blockIdx.x) & 15) < 4) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      T_softmax_maxelem_rf_next_global_local_fragment_1[i] = T_softmax_maxelem_rf_next_global[(((i * 256) + ((((int)threadIdx.x) >> 5) * 64)) + ((int)blockIdx.x))];
    }
    T_softmax_maxelem_local_fragment_1[0] = -CUDART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 16; ++rv) {
      T_softmax_maxelem_local_fragment_1[0] = max(T_softmax_maxelem_local_fragment_1[0], T_softmax_maxelem_rf_next_global_local_fragment_1[rv]);
    }
    T_softmax_maxelem_local_fragment_1[0] = tl::AllReduce<tl::MaxOp, 128, 32, 0, 128>::run_hopper(T_softmax_maxelem_local_fragment_1[0], (&(((float*)buf_dyn_shmem)[0])));
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      *(float4*)(T_batch_matmul_NN_rf_local_fragment_5 + (i_1 * 4)) = *(float4*)(T_batch_matmul_NN_rf + ((((i_1 * 32768) + ((((int)threadIdx.x) >> 5) * 8192)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 31) * 4)));
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 64; ++i_2) {
      reduce_input[i_2] = (T_batch_matmul_NN_rf_local_fragment_5[i_2] * exp2f((T_softmax_maxelem_rf_next_global_local_fragment_1[(i_2 >> 2)] - T_softmax_maxelem_local_fragment_1[0])));
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      T_batch_matmul_NN_local_fragment_1[i_3] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        T_batch_matmul_NN_local_fragment_1[i_3] = (T_batch_matmul_NN_local_fragment_1[i_3] + reduce_input[((rv_1 * 4) + i_3)]);
      }
      T_batch_matmul_NN_local_fragment_1[i_3] = tl::AllReduce<tl::SumOp, 128, 32, 0, 128>::run_hopper(T_batch_matmul_NN_local_fragment_1[i_3], (&(((float*)buf_dyn_shmem)[128])));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      T_softmax_expsum_rf_local_fragment_5[i_4] = T_softmax_expsum_rf[(((i_4 * 256) + ((((int)threadIdx.x) >> 5) * 64)) + ((int)blockIdx.x))];
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      reduce_input_1[i_5] = (T_softmax_expsum_rf_local_fragment_5[i_5] * exp2f((T_softmax_maxelem_rf_next_global_local_fragment_1[i_5] - T_softmax_maxelem_local_fragment_1[0])));
    }
    T_softmax_expsum_local_fragment_1[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
      T_softmax_expsum_local_fragment_1[0] = (T_softmax_expsum_local_fragment_1[0] + reduce_input_1[rv_2]);
    }
    T_softmax_expsum_local_fragment_1[0] = tl::AllReduce<tl::SumOp, 128, 32, 0, 128>::run_hopper(T_softmax_expsum_local_fragment_1[0], (&(((float*)buf_dyn_shmem)[256])));
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      T_softmax_norm_local_fragment_1[i_6] = (T_batch_matmul_NN_local_fragment_1[i_6] / T_softmax_expsum_local_fragment_1[0]);
    }
    uint2 __1;
    float4 v_ = *(float4*)(T_softmax_norm_local_fragment_1 + 0);
    ((half2*)(&(__1.x)))->x = (half_t)(v_.x);
    ((half2*)(&(__1.x)))->y = (half_t)(v_.y);
    ((half2*)(&(__1.y)))->x = (half_t)(v_.z);
    ((half2*)(&(__1.y)))->y = (half_t)(v_.w);
    *(uint2*)(T_cast_pad_local_fragment_1 + 0) = __1;
    *(uint2*)(((half_t*)buf_dyn_shmem) + (((((int)threadIdx.x) & 31) * 4) + 1024)) = *(uint2*)(T_cast_pad_local_fragment_1 + 0);
    tl::fence_proxy_async();
    __syncthreads();
    if (((int)threadIdx.x) == 0) {
      tl::tma_store((&(T_cast[(((((int)blockIdx.x) >> 4) * 512) + ((((int)blockIdx.x) & 15) * 128))])), (&(((half_t*)buf_dyn_shmem)[1024])), 256);
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
    }
  }
}

