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

extern "C" __global__ void main_kernel_kernel(float* __restrict__ T_batch_matmul_NN_rf, float* __restrict__ T_softmax_expsum_rf, float* __restrict__ T_softmax_maxelem_rf, half_t* __restrict__ k, half_t* __restrict__ q, half_t* __restrict__ v);
extern "C" __global__ void main_kernel_1_kernel(float* __restrict__ T_batch_matmul_NN_rf, half_t* __restrict__ T_cast, float* __restrict__ T_softmax_expsum_rf, float* __restrict__ T_softmax_maxelem_rf);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel_kernel(float* __restrict__ T_batch_matmul_NN_rf, float* __restrict__ T_softmax_expsum_rf, float* __restrict__ T_softmax_maxelem_rf, half_t* __restrict__ k, half_t* __restrict__ q, half_t* __restrict__ v) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float T_batch_matmul_NT_local_fragment_1[4];
  float reduce_input[16];
  float reduce_out[4];
  float reduce_out_1[1];
  __shared__ float T_softmax_maxelem_rf_shared_1[1];
  float reduce_input_1[32];
  float reduce_out_2[1];
  float reduce_out_3[1];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    T_batch_matmul_NT_local_fragment_1[i] = 0x0p+0f/*0.000000e+00*/;
  }
  for (int k_0 = 0; k_0 < 2; ++k_0) {
    __syncthreads();
    if (((int)threadIdx.x) < 64) {
      #pragma unroll
      for (int i_1 = 0; i_1 < 1; ++i_1) {
        ((half_t*)buf_dyn_shmem)[(((int)threadIdx.x) + 2048)] = q[((((((int)blockIdx.z) * 1024) + (((int)blockIdx.y) * 128)) + (k_0 * 64)) + ((int)threadIdx.x))];
      }
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      *(uint4*)(((half_t*)buf_dyn_shmem) + ((i_2 * 1024) + (((int)threadIdx.x) * 8))) = *(uint4*)(k + ((((((((int)blockIdx.z) * 65536) + (((int)blockIdx.x) * 4096)) + (i_2 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + (k_0 * 64)) + ((((int)threadIdx.x) & 7) * 8)));
    }
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      float4 __1;
        float4 __2;
        uint2 v_ = *(uint2*)(((half_t*)buf_dyn_shmem) + (((((int)threadIdx.x) & 15) * 4) + 2048));
        __2.x = (float)(((half2*)(&(v_.x)))->x);
        __2.y = (float)(((half2*)(&(v_.x)))->y);
        __2.z = (float)(((half2*)(&(v_.y)))->x);
        __2.w = (float)(((half2*)(&(v_.y)))->y);
        float4 __3;
        uint2 v__1 = *(uint2*)(((half_t*)buf_dyn_shmem) + ((i_3 * 512) + (((int)threadIdx.x) * 4)));
        __3.x = (float)(((half2*)(&(v__1.x)))->x);
        __3.y = (float)(((half2*)(&(v__1.x)))->y);
        __3.z = (float)(((half2*)(&(v__1.y)))->x);
        __3.w = (float)(((half2*)(&(v__1.y)))->y);
        __1.x = (__2.x*__3.x);
        __1.y = (__2.y*__3.y);
        __1.z = (__2.z*__3.z);
        __1.w = (__2.w*__3.w);
      *(float4*)(reduce_input + (i_3 * 4)) = __1;
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      reduce_out[i_4] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv = 0; rv < 4; ++rv) {
        reduce_out[i_4] = (reduce_out[i_4] + reduce_input[((i_4 * 4) + rv)]);
      }
      reduce_out[i_4] = tl::AllReduce<tl::SumOp, 16, 1, 0, 128>::run_hopper(reduce_out[i_4]);
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      T_batch_matmul_NT_local_fragment_1[i_5] = (T_batch_matmul_NT_local_fragment_1[i_5] + reduce_out[i_5]);
    }
  }
  #pragma unroll
  for (int i_6 = 0; i_6 < 4; ++i_6) {
    float condval;
    if (((((((int)blockIdx.x) * 32) + (i_6 * 8)) + (((int)threadIdx.x) >> 4)) <= (((int)blockIdx.y) + 504))) {
      condval = (T_batch_matmul_NT_local_fragment_1[i_6] * 0x1.0527dbd87e24dp-3f/*1.275174e-01*/);
    } else {
      condval = -0x1.fffffep+127f/*-3.402823e+38*/;
    }
    T_batch_matmul_NT_local_fragment_1[i_6] = condval;
  }
  reduce_out_1[0] = -CUDART_INF_F;
  #pragma unroll
  for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
    reduce_out_1[0] = max(reduce_out_1[0], T_batch_matmul_NT_local_fragment_1[rv_1]);
  }
  __syncthreads();
  reduce_out_1[0] = tl::AllReduce<tl::MaxOp, 128, 16, 0, 128>::run_hopper(reduce_out_1[0], (&(((float*)buf_dyn_shmem)[0])));
  #pragma unroll
  for (int i_7 = 0; i_7 < 1; ++i_7) {
    T_softmax_maxelem_rf_shared_1[0] = reduce_out_1[0];
  }
  #pragma unroll
  for (int i_8 = 0; i_8 < 4; ++i_8) {
    T_batch_matmul_NT_local_fragment_1[i_8] = exp2f((T_batch_matmul_NT_local_fragment_1[i_8] - T_softmax_maxelem_rf_shared_1[0]));
  }
  if ((((int)threadIdx.x) % 16) == 0) {
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      ((half_t*)buf_dyn_shmem)[(((i_9 * 8) + (((int)threadIdx.x) >> 4)) + 2048)] = ((half_t)T_batch_matmul_NT_local_fragment_1[i_9]);
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_10 = 0; i_10 < 32; ++i_10) {
    reduce_input_1[i_10] = (((float)((half_t*)buf_dyn_shmem)[(i_10 + 2048)]) * ((float)v[((((((int)blockIdx.z) * 65536) + (((int)blockIdx.x) * 4096)) + (i_10 * 128)) + ((int)threadIdx.x))]));
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 1; ++i_11) {
    reduce_out_2[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 32; ++rv_2) {
      reduce_out_2[0] = (reduce_out_2[0] + reduce_input_1[rv_2]);
    }
  }
  #pragma unroll
  for (int i_12 = 0; i_12 < 1; ++i_12) {
    T_batch_matmul_NN_rf[((((((int)blockIdx.x) * 4096) + (((int)blockIdx.z) * 1024)) + (((int)blockIdx.y) * 128)) + ((int)threadIdx.x))] = reduce_out_2[0];
  }
  reduce_out_3[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_3 = 0; rv_3 < 4; ++rv_3) {
    reduce_out_3[0] = (reduce_out_3[0] + T_batch_matmul_NT_local_fragment_1[rv_3]);
  }
  __syncthreads();
  reduce_out_3[0] = tl::AllReduce<tl::SumOp, 128, 16, 0, 128>::run_hopper(reduce_out_3[0], (&(((float*)buf_dyn_shmem)[0])));
  T_softmax_expsum_rf[(((((int)blockIdx.x) * 32) + (((int)blockIdx.z) * 8)) + ((int)blockIdx.y))] = reduce_out_3[0];
  T_softmax_maxelem_rf[(((((int)blockIdx.x) * 32) + (((int)blockIdx.z) * 8)) + ((int)blockIdx.y))] = T_softmax_maxelem_rf_shared_1[0];
}

extern "C" __global__ void __launch_bounds__(128, 1) main_kernel_1_kernel(float* __restrict__ T_batch_matmul_NN_rf, half_t* __restrict__ T_cast, float* __restrict__ T_softmax_expsum_rf, float* __restrict__ T_softmax_maxelem_rf) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float T_softmax_maxelem_rf_local_fragment_1[4];
  float T_softmax_maxelem_local_fragment_1[1];
  float T_batch_matmul_NN_rf_local_fragment_1[16];
  float T_batch_matmul_NN_local_fragment_1[4];
  float T_softmax_expsum_rf_local_fragment_1[4];
  float T_softmax_expsum_local_fragment_1[1];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    T_softmax_maxelem_rf_local_fragment_1[i] = T_softmax_maxelem_rf[(((i * 128) + ((((int)threadIdx.x) >> 5) * 32)) + ((int)blockIdx.x))];
  }
  T_softmax_maxelem_local_fragment_1[0] = -CUDART_INF_F;
  #pragma unroll
  for (int rv = 0; rv < 4; ++rv) {
    T_softmax_maxelem_local_fragment_1[0] = max(T_softmax_maxelem_local_fragment_1[0], T_softmax_maxelem_rf_local_fragment_1[rv]);
  }
  T_softmax_maxelem_local_fragment_1[0] = tl::AllReduce<tl::MaxOp, 128, 32, 0, 128>::run_hopper(T_softmax_maxelem_local_fragment_1[0], (&(((float*)buf_dyn_shmem)[0])));
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    *(float4*)(T_batch_matmul_NN_rf_local_fragment_1 + (i_1 * 4)) = *(float4*)(T_batch_matmul_NN_rf + ((((i_1 * 16384) + ((((int)threadIdx.x) >> 5) * 4096)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 31) * 4)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    T_batch_matmul_NN_rf_local_fragment_1[i_2] = (T_batch_matmul_NN_rf_local_fragment_1[i_2] * exp2f((T_softmax_maxelem_rf_local_fragment_1[(i_2 >> 2)] - T_softmax_maxelem_local_fragment_1[0])));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    T_batch_matmul_NN_local_fragment_1[i_3] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
      T_batch_matmul_NN_local_fragment_1[i_3] = (T_batch_matmul_NN_local_fragment_1[i_3] + T_batch_matmul_NN_rf_local_fragment_1[((rv_1 * 4) + i_3)]);
    }
    T_batch_matmul_NN_local_fragment_1[i_3] = tl::AllReduce<tl::SumOp, 128, 32, 0, 128>::run_hopper(T_batch_matmul_NN_local_fragment_1[i_3], (&(((float*)buf_dyn_shmem)[0])));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    T_softmax_expsum_rf_local_fragment_1[i_4] = T_softmax_expsum_rf[(((i_4 * 128) + ((((int)threadIdx.x) >> 5) * 32)) + ((int)blockIdx.x))];
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 4; ++i_5) {
    T_softmax_expsum_rf_local_fragment_1[i_5] = (T_softmax_expsum_rf_local_fragment_1[i_5] * exp2f((T_softmax_maxelem_rf_local_fragment_1[i_5] - T_softmax_maxelem_local_fragment_1[0])));
  }
  T_softmax_expsum_local_fragment_1[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_2 = 0; rv_2 < 4; ++rv_2) {
    T_softmax_expsum_local_fragment_1[0] = (T_softmax_expsum_local_fragment_1[0] + T_softmax_expsum_rf_local_fragment_1[rv_2]);
  }
  T_softmax_expsum_local_fragment_1[0] = tl::AllReduce<tl::SumOp, 128, 32, 0, 128>::run_hopper(T_softmax_expsum_local_fragment_1[0], (&(((float*)buf_dyn_shmem)[0])));
  #pragma unroll
  for (int i_6 = 0; i_6 < 4; ++i_6) {
    T_batch_matmul_NN_local_fragment_1[i_6] = (T_batch_matmul_NN_local_fragment_1[i_6] / T_softmax_expsum_local_fragment_1[0]);
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    uint2 __1;
    float4 v_ = *(float4*)(T_batch_matmul_NN_local_fragment_1 + 0);
    ((half2*)(&(__1.x)))->x = (half_t)(v_.x);
    ((half2*)(&(__1.x)))->y = (half_t)(v_.y);
    ((half2*)(&(__1.y)))->x = (half_t)(v_.z);
    ((half2*)(&(__1.y)))->y = (half_t)(v_.w);
    *(uint2*)(((half_t*)buf_dyn_shmem) + ((((int)threadIdx.x) & 31) * 4)) = __1;
  }
  __syncthreads();
  #pragma unroll
  for (int i_7 = 0; i_7 < 1; ++i_7) {
    T_cast[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = ((half_t*)buf_dyn_shmem)[((int)threadIdx.x)];
  }
}

