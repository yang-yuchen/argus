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
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel_kernel(float* __restrict__ T_batch_matmul_NN_rf, float* __restrict__ T_softmax_expsum_rf, float* __restrict__ T_softmax_maxelem_rf, half_t* __restrict__ k, half_t* __restrict__ q, half_t* __restrict__ v) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float T_batch_matmul_NT_local_fragment_1[8];
  float reduce_input[32];
  float reduce_out[8];
  float T_score_mod_local_fragment_1[8];
  float reduce_out_1[1];
  __shared__ float T_softmax_maxelem_rf_shared_1[1];
  float T_softmax_exp_local_fragment_1[8];
  float reduce_input_1[32];
  float reduce_out_2[1];
  float reduce_out_3[1];
  __shared__ uint64_t mbarrier_mem[1];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(128);
  }
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    if (tl::tl_shuffle_elect<128>()) {
      mbarrier[0].expect_transaction(256);
      tl::tma_load((&(((half_t*)buf_dyn_shmem)[4096])), (&(q[((((int)blockIdx.z) * 512) + (((int)blockIdx.y) * 128))])), mbarrier[0], 256);
      mbarrier[0].expect_transaction(8192);
      tl::tma_load((&(((half_t*)buf_dyn_shmem)[0])), (&(k[((((int)blockIdx.z) * 32768) + (((int)blockIdx.x) * 4096))])), mbarrier[0], 8192);
    }
    mbarrier[0].arrive();
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      T_batch_matmul_NT_local_fragment_1[i] = 0x0p+0f/*0.000000e+00*/;
    }
    tl::fence_proxy_async();
    mbarrier[0].wait(0);
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      float4 __1;
        float4 __2;
        uint2 v_ = *(uint2*)(((half_t*)buf_dyn_shmem) + (((((int)threadIdx.x) & 31) * 4) + 4096));
        __2.x = (float)(((half2*)(&(v_.x)))->x);
        __2.y = (float)(((half2*)(&(v_.x)))->y);
        __2.z = (float)(((half2*)(&(v_.y)))->x);
        __2.w = (float)(((half2*)(&(v_.y)))->y);
        float4 __3;
        uint2 v__1 = *(uint2*)(((half_t*)buf_dyn_shmem) + ((i_1 * 512) + (((int)threadIdx.x) * 4)));
        __3.x = (float)(((half2*)(&(v__1.x)))->x);
        __3.y = (float)(((half2*)(&(v__1.x)))->y);
        __3.z = (float)(((half2*)(&(v__1.y)))->x);
        __3.w = (float)(((half2*)(&(v__1.y)))->y);
        __1.x = (__2.x*__3.x);
        __1.y = (__2.y*__3.y);
        __1.z = (__2.z*__3.z);
        __1.w = (__2.w*__3.w);
      *(float4*)(reduce_input + (i_1 * 4)) = __1;
    }
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int i_2 = 0; i_2 < 8; ++i_2) {
      reduce_out[i_2] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv = 0; rv < 4; ++rv) {
        reduce_out[i_2] = (reduce_out[i_2] + reduce_input[((i_2 * 4) + rv)]);
      }
      reduce_out[i_2] = tl::AllReduce<tl::SumOp, 32, 1, 0, 128>::run_hopper(reduce_out[i_2], (&(((float*)buf_dyn_shmem)[0])));
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      T_batch_matmul_NT_local_fragment_1[i_3] = (T_batch_matmul_NT_local_fragment_1[i_3] + reduce_out[i_3]);
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      float condval;
      if (((((((int)blockIdx.x) * 32) + (i_4 * 4)) + (((int)threadIdx.x) >> 5)) <= (((int)blockIdx.y) + 252))) {
        condval = (T_batch_matmul_NT_local_fragment_1[i_4] * 0x1.0527dbd87e24dp-3f/*1.275174e-01*/);
      } else {
        condval = -0x1.fffffep+127f/*-3.402823e+38*/;
      }
      T_score_mod_local_fragment_1[i_4] = condval;
    }
    reduce_out_1[0] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
      reduce_out_1[0] = max(reduce_out_1[0], T_score_mod_local_fragment_1[rv_1]);
    }
    reduce_out_1[0] = tl::AllReduce<tl::MaxOp, 128, 32, 0, 128>::run_hopper(reduce_out_1[0], (&(((float*)buf_dyn_shmem)[0])));
    T_softmax_maxelem_rf_shared_1[0] = reduce_out_1[0];
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      T_softmax_exp_local_fragment_1[i_5] = exp2f((T_score_mod_local_fragment_1[i_5] - T_softmax_maxelem_rf_shared_1[0]));
    }
    if ((((int)threadIdx.x) % 32) == 0) {
      #pragma unroll
      for (int i_6 = 0; i_6 < 8; ++i_6) {
        ((half_t*)buf_dyn_shmem)[(((i_6 * 4) + (((int)threadIdx.x) >> 5)) + 4096)] = ((half_t)T_softmax_exp_local_fragment_1[i_6]);
      }
    }
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int i_7 = 0; i_7 < 32; ++i_7) {
      reduce_input_1[i_7] = (((float)((half_t*)buf_dyn_shmem)[(i_7 + 4096)]) * ((float)v[((((((int)blockIdx.z) * 32768) + (((int)blockIdx.x) * 4096)) + (i_7 * 128)) + ((int)threadIdx.x))]));
    }
    reduce_out_2[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 32; ++rv_2) {
      reduce_out_2[0] = (reduce_out_2[0] + reduce_input_1[rv_2]);
    }
    T_batch_matmul_NN_rf[((((((int)blockIdx.x) * 2048) + (((int)blockIdx.z) * 512)) + (((int)blockIdx.y) * 128)) + ((int)threadIdx.x))] = reduce_out_2[0];
    reduce_out_3[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 8; ++rv_3) {
      reduce_out_3[0] = (reduce_out_3[0] + T_softmax_exp_local_fragment_1[rv_3]);
    }
    tl::__sync_thread_partial<3, 128>();
    reduce_out_3[0] = tl::AllReduce<tl::SumOp, 128, 32, 0, 128>::run_hopper(reduce_out_3[0], (&(((float*)buf_dyn_shmem)[0])));
    T_softmax_expsum_rf[(((((int)blockIdx.x) * 16) + (((int)blockIdx.z) * 4)) + ((int)blockIdx.y))] = reduce_out_3[0];
    T_softmax_maxelem_rf[(((((int)blockIdx.x) * 16) + (((int)blockIdx.z) * 4)) + ((int)blockIdx.y))] = T_softmax_maxelem_rf_shared_1[0];
  }
}

extern "C" __global__ void __launch_bounds__(64, 1) main_kernel_1_kernel(float* __restrict__ T_batch_matmul_NN_rf, half_t* __restrict__ T_cast, float* __restrict__ T_softmax_expsum_rf, float* __restrict__ T_softmax_maxelem_rf) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float T_softmax_maxelem_rf_local_fragment_1[4];
  float T_softmax_maxelem_local_fragment_1[1];
  float T_batch_matmul_NN_rf_local_fragment_1[16];
  float T_batch_matmul_NN_local_fragment_1[4];
  float T_softmax_expsum_rf_local_fragment_1[4];
  float T_softmax_expsum_local_fragment_1[1];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    T_softmax_maxelem_rf_local_fragment_1[i] = T_softmax_maxelem_rf[(((i * 32) + ((((int)threadIdx.x) >> 5) * 16)) + ((int)blockIdx.x))];
  }
  T_softmax_maxelem_local_fragment_1[0] = -CUDART_INF_F;
  #pragma unroll
  for (int rv = 0; rv < 4; ++rv) {
    T_softmax_maxelem_local_fragment_1[0] = max(T_softmax_maxelem_local_fragment_1[0], T_softmax_maxelem_rf_local_fragment_1[rv]);
  }
  T_softmax_maxelem_local_fragment_1[0] = tl::AllReduce<tl::MaxOp, 64, 32, 0, 64>::run_hopper(T_softmax_maxelem_local_fragment_1[0], (&(((float*)buf_dyn_shmem)[0])));
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    *(float4*)(T_batch_matmul_NN_rf_local_fragment_1 + (i_1 * 4)) = *(float4*)(T_batch_matmul_NN_rf + ((((i_1 * 4096) + ((((int)threadIdx.x) >> 5) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 31) * 4)));
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
    T_batch_matmul_NN_local_fragment_1[i_3] = tl::AllReduce<tl::SumOp, 64, 32, 0, 64>::run_hopper(T_batch_matmul_NN_local_fragment_1[i_3], (&(((float*)buf_dyn_shmem)[0])));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    T_softmax_expsum_rf_local_fragment_1[i_4] = T_softmax_expsum_rf[(((i_4 * 32) + ((((int)threadIdx.x) >> 5) * 16)) + ((int)blockIdx.x))];
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
  T_softmax_expsum_local_fragment_1[0] = tl::AllReduce<tl::SumOp, 64, 32, 0, 64>::run_hopper(T_softmax_expsum_local_fragment_1[0], (&(((float*)buf_dyn_shmem)[0])));
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
  tl::fence_proxy_async();
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    tl::tma_store((&(T_cast[(((int)blockIdx.x) * 128)])), (&(((half_t*)buf_dyn_shmem)[0])), 256);
    tl::tma_store_arrive();
    tl::tma_store_wait<0>();
  }
}

