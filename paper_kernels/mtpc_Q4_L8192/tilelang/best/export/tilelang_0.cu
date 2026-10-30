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

extern "C" __global__ void main_kernel_kernel(float* __restrict__ T_batch_matmul_NN_rf, float* __restrict__ T_softmax_expsum_rf, float* __restrict__ T_softmax_maxelem_rf_next_global, half_t* __restrict__ k, half_t* __restrict__ q, half_t* __restrict__ v);
extern "C" __global__ void main_kernel_1_kernel(float* __restrict__ T_batch_matmul_NN_rf, half_t* __restrict__ T_cast, float* __restrict__ T_softmax_expsum_rf, float* __restrict__ T_softmax_maxelem_rf_next_global);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel_kernel(float* __restrict__ T_batch_matmul_NN_rf, float* __restrict__ T_softmax_expsum_rf, float* __restrict__ T_softmax_maxelem_rf_next_global, half_t* __restrict__ k, half_t* __restrict__ q, half_t* __restrict__ v) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float T_softmax_maxelem_rf_prev_local_fragment_1[2];
  float T_batch_matmul_NN_rf_local_fragment_1[8];
  float T_softmax_expsum_rf_local_fragment_1[2];
  float T_softmax_maxelem_rf_next_local_fragment_1[2];
  float T_softmax_maxelem_rf_next_local_fragment[2];
  float T_batch_matmul_NT_local_fragment_1[4];
  float T_softmax_maxelem_rf_rf_local_fragment_1[2];
  float T_softmax_expsum_rf_rf_local_fragment_1[2];
  float T_batch_matmul_NN_rf_rf_local_fragment_1[8];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    T_softmax_maxelem_rf_prev_local_fragment_1[i] = -0x1.fffffep+127f/*-3.402823e+38*/;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    *(float2*)(T_batch_matmul_NN_rf_local_fragment_1 + (i_1 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    T_softmax_expsum_rf_local_fragment_1[i_2] = 0x0p+0f/*0.000000e+00*/;
  }
  uint4 condval;
  if ((((int)threadIdx.x) < 64)) {
    condval = *(uint4*)(q + ((((int)blockIdx.y) * 512) + (((int)threadIdx.x) * 8)));
  } else {
    condval = make_uint4(__pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)));
  }
  *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 1024) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)) = condval;
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    T_softmax_maxelem_rf_next_local_fragment_1[i_3] = T_softmax_maxelem_rf_next_local_fragment[i_3];
  }
  for (int j_0_1 = 0; j_0_1 < 2; ++j_0_1) {
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      *(float2*)(T_batch_matmul_NT_local_fragment_1 + (i_4 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(k + (((((((int)blockIdx.y) * 1048576) + (((int)blockIdx.x) * 16384)) + (j_0_1 * 8192)) + (i_5 * 2048)) + (((int)threadIdx.x) * 8)));
    }
    tl::fence_proxy_async();
    __syncthreads();
    tl::gemm_ss<16, 64, 128, 1, 8, 0, 1, 0, 128, 128, 0, 0, false>((&(((half_t*)buf_dyn_shmem)[16384])), (&(((half_t*)buf_dyn_shmem)[0])), (&(T_batch_matmul_NT_local_fragment_1[0])));
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      float condval_1;
      if (((((((((int)blockIdx.x) * 128) + (j_0_1 * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1)) <= ((((i_6 >> 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 8188))) {
        condval_1 = (T_batch_matmul_NT_local_fragment_1[i_6] * 0x1.0527dbd87e24dp-3f/*1.275174e-01*/);
      } else {
        condval_1 = -0x1.fffffep+127f/*-3.402823e+38*/;
      }
      T_batch_matmul_NT_local_fragment_1[i_6] = condval_1;
    }
    __syncthreads();
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      T_softmax_maxelem_rf_rf_local_fragment_1[i_7] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 2; ++rv) {
        T_softmax_maxelem_rf_rf_local_fragment_1[i_7] = max(T_softmax_maxelem_rf_rf_local_fragment_1[i_7], T_batch_matmul_NT_local_fragment_1[((i_7 * 2) + rv)]);
      }
      T_softmax_maxelem_rf_rf_local_fragment_1[i_7] = tl::AllReduce<tl::MaxOp, 256, 32, 0, 256>::run_hopper(T_softmax_maxelem_rf_rf_local_fragment_1[i_7], (&(((float*)buf_dyn_shmem)[9728])));
      T_softmax_maxelem_rf_rf_local_fragment_1[i_7] = tl::AllReduce<tl::MaxOp, 4, 1, 0, 256>::run_hopper(T_softmax_maxelem_rf_rf_local_fragment_1[i_7]);
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      T_softmax_maxelem_rf_next_local_fragment_1[i_8] = max(T_softmax_maxelem_rf_prev_local_fragment_1[i_8], T_softmax_maxelem_rf_rf_local_fragment_1[i_8]);
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      T_batch_matmul_NT_local_fragment_1[i_9] = exp2f((T_batch_matmul_NT_local_fragment_1[i_9] - T_softmax_maxelem_rf_next_local_fragment_1[(i_9 >> 1)]));
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      T_softmax_expsum_rf_rf_local_fragment_1[i_10] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
        T_softmax_expsum_rf_rf_local_fragment_1[i_10] = (T_softmax_expsum_rf_rf_local_fragment_1[i_10] + T_batch_matmul_NT_local_fragment_1[((i_10 * 2) + rv_1)]);
      }
      T_softmax_expsum_rf_rf_local_fragment_1[i_10] = tl::AllReduce<tl::SumOp, 256, 32, 0, 256>::run_hopper(T_softmax_expsum_rf_rf_local_fragment_1[i_10], (&(((float*)buf_dyn_shmem)[9984])));
      T_softmax_expsum_rf_rf_local_fragment_1[i_10] = tl::AllReduce<tl::SumOp, 4, 1, 0, 256>::run_hopper(T_softmax_expsum_rf_rf_local_fragment_1[i_10]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 1; ++i_11) {
      tl::ptx_stmatrix_x2((&(((half_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 18432)])), __pack_half2(((half_t)T_batch_matmul_NT_local_fragment_1[0]), ((half_t)T_batch_matmul_NT_local_fragment_1[1])), __pack_half2(((half_t)T_batch_matmul_NT_local_fragment_1[2]), ((half_t)T_batch_matmul_NT_local_fragment_1[3])));
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 4; ++i_12) {
      *(uint4*)(((half_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_12 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)) = *(uint4*)(v + (((((((int)blockIdx.y) * 1048576) + (((int)blockIdx.x) * 16384)) + (j_0_1 * 8192)) + (i_12 * 2048)) + (((int)threadIdx.x) * 8)));
    }
    tl::fence_proxy_async();
    __syncthreads();
    tl::gemm_ss<16, 128, 64, 1, 8, 0, 0, 1, 64, 128, 0, 0, false>((&(((half_t*)buf_dyn_shmem)[18432])), (&(((half_t*)buf_dyn_shmem)[8192])), (&(T_batch_matmul_NN_rf_rf_local_fragment_1[0])));
    #pragma unroll
    for (int i_13 = 0; i_13 < 8; ++i_13) {
      T_batch_matmul_NN_rf_local_fragment_1[i_13] = ((T_batch_matmul_NN_rf_local_fragment_1[i_13] * exp2f((T_softmax_maxelem_rf_prev_local_fragment_1[((i_13 & 3) >> 1)] - T_softmax_maxelem_rf_next_local_fragment_1[((i_13 & 3) >> 1)]))) + T_batch_matmul_NN_rf_rf_local_fragment_1[i_13]);
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      T_softmax_expsum_rf_local_fragment_1[i_14] = ((T_softmax_expsum_rf_local_fragment_1[i_14] * exp2f((T_softmax_maxelem_rf_prev_local_fragment_1[i_14] - T_softmax_maxelem_rf_next_local_fragment_1[i_14]))) + T_softmax_expsum_rf_rf_local_fragment_1[i_14]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      T_softmax_maxelem_rf_prev_local_fragment_1[i_15] = T_softmax_maxelem_rf_next_local_fragment_1[i_15];
    }
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 4; ++i_16) {
    *(float2*)(T_batch_matmul_NN_rf + (((((((((int)blockIdx.x) * 8192) + (((int)blockIdx.y) * 2048)) + ((i_16 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_16 >> 1) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(T_batch_matmul_NN_rf_local_fragment_1 + (i_16 * 2));
  }
  #pragma unroll
  for (int i_17 = 0; i_17 < 2; ++i_17) {
    T_softmax_expsum_rf[((((((int)blockIdx.x) * 64) + (((int)blockIdx.y) * 16)) + (i_17 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = T_softmax_expsum_rf_local_fragment_1[i_17];
  }
  #pragma unroll
  for (int i_18 = 0; i_18 < 2; ++i_18) {
    T_softmax_maxelem_rf_next_global[((((((int)blockIdx.x) * 64) + (((int)blockIdx.y) * 16)) + (i_18 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = T_softmax_maxelem_rf_next_local_fragment_1[i_18];
  }
}

extern "C" __global__ void __launch_bounds__(64, 1) main_kernel_1_kernel(float* __restrict__ T_batch_matmul_NN_rf, half_t* __restrict__ T_cast, float* __restrict__ T_softmax_expsum_rf, float* __restrict__ T_softmax_maxelem_rf_next_global) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float T_softmax_maxelem_rf_next_global_local_fragment_1[32];
  float T_softmax_maxelem_local_fragment_1[1];
  float T_batch_matmul_NN_rf_local_fragment_5[128];
  float reduce_input[128];
  float T_batch_matmul_NN_local_fragment_1[4];
  float T_softmax_expsum_rf_local_fragment_5[32];
  float reduce_input_1[32];
  float T_softmax_expsum_local_fragment_1[1];
  float T_softmax_norm_local_fragment_1[4];
  half_t T_cast_pad_local_fragment_1[4];
  if ((((int)blockIdx.x) & 15) < 4) {
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      T_softmax_maxelem_rf_next_global_local_fragment_1[i] = T_softmax_maxelem_rf_next_global[(((i * 128) + ((((int)threadIdx.x) >> 5) * 64)) + ((int)blockIdx.x))];
    }
    T_softmax_maxelem_local_fragment_1[0] = -CUDART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 32; ++rv) {
      T_softmax_maxelem_local_fragment_1[0] = max(T_softmax_maxelem_local_fragment_1[0], T_softmax_maxelem_rf_next_global_local_fragment_1[rv]);
    }
    T_softmax_maxelem_local_fragment_1[0] = tl::AllReduce<tl::MaxOp, 64, 32, 0, 64>::run_hopper(T_softmax_maxelem_local_fragment_1[0], (&(((float*)buf_dyn_shmem)[0])));
    #pragma unroll
    for (int i_1 = 0; i_1 < 32; ++i_1) {
      *(float4*)(T_batch_matmul_NN_rf_local_fragment_5 + (i_1 * 4)) = *(float4*)(T_batch_matmul_NN_rf + ((((i_1 * 16384) + ((((int)threadIdx.x) >> 5) * 8192)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 31) * 4)));
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 128; ++i_2) {
      reduce_input[i_2] = (T_batch_matmul_NN_rf_local_fragment_5[i_2] * exp2f((T_softmax_maxelem_rf_next_global_local_fragment_1[(i_2 >> 2)] - T_softmax_maxelem_local_fragment_1[0])));
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      T_batch_matmul_NN_local_fragment_1[i_3] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 32; ++rv_1) {
        T_batch_matmul_NN_local_fragment_1[i_3] = (T_batch_matmul_NN_local_fragment_1[i_3] + reduce_input[((rv_1 * 4) + i_3)]);
      }
      T_batch_matmul_NN_local_fragment_1[i_3] = tl::AllReduce<tl::SumOp, 64, 32, 0, 64>::run_hopper(T_batch_matmul_NN_local_fragment_1[i_3], (&(((float*)buf_dyn_shmem)[64])));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 32; ++i_4) {
      T_softmax_expsum_rf_local_fragment_5[i_4] = T_softmax_expsum_rf[(((i_4 * 128) + ((((int)threadIdx.x) >> 5) * 64)) + ((int)blockIdx.x))];
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 32; ++i_5) {
      reduce_input_1[i_5] = (T_softmax_expsum_rf_local_fragment_5[i_5] * exp2f((T_softmax_maxelem_rf_next_global_local_fragment_1[i_5] - T_softmax_maxelem_local_fragment_1[0])));
    }
    T_softmax_expsum_local_fragment_1[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 32; ++rv_2) {
      T_softmax_expsum_local_fragment_1[0] = (T_softmax_expsum_local_fragment_1[0] + reduce_input_1[rv_2]);
    }
    T_softmax_expsum_local_fragment_1[0] = tl::AllReduce<tl::SumOp, 64, 32, 0, 64>::run_hopper(T_softmax_expsum_local_fragment_1[0], (&(((float*)buf_dyn_shmem)[128])));
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
    *(uint2*)(((half_t*)buf_dyn_shmem) + (((((int)threadIdx.x) & 31) * 4) + 384)) = *(uint2*)(T_cast_pad_local_fragment_1 + 0);
    __syncthreads();
    *(uint1*)(T_cast + ((((((int)blockIdx.x) >> 4) * 512) + ((((int)blockIdx.x) & 15) * 128)) + (((int)threadIdx.x) * 2))) = *(uint1*)(((half_t*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 384));
  }
}

