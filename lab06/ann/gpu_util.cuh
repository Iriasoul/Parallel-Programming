// gpu_util.cuh 
// GPU 工具函数：错误检查宏、事件计时、公共 kernel
// 所有 .cuh 共用这里的 kernel
#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do{ cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    std::fprintf(stderr,"[CUDA] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); std::exit(1);} }while(0)
#define CUBLAS_CHECK(call) do{ cublasStatus_t _s=(call); if(_s!=CUBLAS_STATUS_SUCCESS){ \
    std::fprintf(stderr,"[cuBLAS] %s:%d status %d\n",__FILE__,__LINE__,(int)_s); std::exit(1);} }while(0)

struct GpuTimer {
    cudaEvent_t a, b;
    GpuTimer(){ cudaEventCreate(&a); cudaEventCreate(&b); }
    ~GpuTimer(){ cudaEventDestroy(a); cudaEventDestroy(b); }
    void  start(){ cudaEventRecord(a, 0); }
    float stop_ms(){ cudaEventRecord(b, 0); cudaEventSynchronize(b);
                     float ms = 0; cudaEventElapsedTime(&ms, a, b); return ms; }
};

// 内积距离：dis = 1 - ip
__global__ void ip_to_dist(float* S, size_t total) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < total) S[i] = 1.0f - S[i];
}

// 每 block 处理一行(长度 N)，选最小的 topk 个下标；blockDim.x 为 2 的幂
// out_dist 传 nullptr 则只写下标。每轮归约取全局最小并置 FLT_MAX。
__global__ void topk_rows(float* dist, int N, int topk, uint32_t* out_idx, float* out_dist) {
    int q = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    float* row = dist + (size_t)q * N;
    extern __shared__ char smem[];
    float* sv = reinterpret_cast<float*>(smem);
    int*   si = reinterpret_cast<int*>(sv + nt);
    for (int it = 0; it < topk; ++it) {
        float best = FLT_MAX; int bi = 0;
        for (int j = tid; j < N; j += nt) { float v = row[j]; if (v < best) { best = v; bi = j; } }
        sv[tid] = best; si[tid] = bi; __syncthreads();
        for (int s = nt >> 1; s > 0; s >>= 1) {
            if (tid < s && sv[tid + s] < sv[tid]) { sv[tid] = sv[tid + s]; si[tid] = si[tid + s]; }
            __syncthreads();
        }
        if (tid == 0) {
            out_idx[(size_t)q * topk + it] = (uint32_t)si[0];
            if (out_dist) out_dist[(size_t)q * topk + it] = sv[0];
            row[si[0]] = FLT_MAX;
        }
        __syncthreads();
    }
}
