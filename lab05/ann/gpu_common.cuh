#pragma once

#include "common.cuh"
#include <cfloat>

// 内积距离
__global__ void ip_to_dist_kernel(float* S, size_t total) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < total) S[i] = 1.0f - S[i];
}

// 块级 top-k：每个 block 处理一行 (长度 N)，输出该行最小的 k 个 (idx,dist)。
// 做 k 次"块内归约求全局最小 + 标记 +inf" // 要求 blockDim.x 为 2 的幂
// 会就地修改 dist (把选中位置置 +inf)
__global__ void topk_kernel(float* dist, int N, int k, uint32_t* out_idx, float* out_dist) {
    int q = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    float* row = dist + (size_t)q * N;
    extern __shared__ char smem[];
    float* sv = reinterpret_cast<float*>(smem);
    int*   si = reinterpret_cast<int*>(sv + nt);
    for (int it = 0; it < k; ++it) {
        float best = FLT_MAX; int bi = 0;
        for (int j = tid; j < N; j += nt) { float v = row[j]; if (v < best) { best = v; bi = j; } }
        sv[tid] = best; si[tid] = bi; __syncthreads();
        for (int s = nt >> 1; s > 0; s >>= 1) {
            if (tid < s && sv[tid+s] < sv[tid]) { sv[tid] = sv[tid+s]; si[tid] = si[tid+s]; }
            __syncthreads();
        }
        if (tid == 0) {
            out_idx[(size_t)q*k + it] = (uint32_t)si[0];
            out_dist[(size_t)q*k + it] = sv[0];
            row[si[0]] = FLT_MAX;
        }
        __syncthreads();
    }
}

// 按索引把若干行 gather 到连续缓冲 
__global__ void gather_rows(const float* src, float* dst, const int* gidx, int n, int d) {
    int i = blockIdx.x; if (i >= n) return;
    int s = gidx[i];
    for (int j = threadIdx.x; j < d; j += blockDim.x) dst[(size_t)i*d + j] = src[(size_t)s*d + j];
}

// PQ：算查找表 LUT[i][j][t] 
__global__ void build_lut(const float* Q, const float* codebook, float* LUT,
                          int B, int m, int ksub, int subdim) {
    long idx = blockIdx.x*(long)blockDim.x + threadIdx.x;
    long total = (long)B*m*ksub; if (idx >= total) return;
    int t = (int)(idx % ksub); int j = (int)((idx/ksub)%m); int i = (int)(idx/((long)ksub*m));
    const float* qsub = Q + (long)i*(m*subdim) + (long)j*subdim;
    const float* csub = codebook + ((long)j*ksub + t)*subdim;
    float s = 0; for (int x = 0; x < subdim; ++x) s += qsub[x]*csub[x];
    LUT[idx] = s;
}

// PQ-ADC
__global__ void adc_kernel(const uint8_t* codes_c, const float* LUT, float* Sc,
                           int B, int vc, int m, int ksub) {
    long idx = blockIdx.x*(long)blockDim.x + threadIdx.x;
    long total = (long)B*vc; if (idx >= total) return;
    int v = (int)(idx % vc); int i = (int)(idx / vc);
    const uint8_t* code = codes_c + (long)v*m;
    const float* lut_i = LUT + (long)i*m*ksub;
    float ip = 0; for (int j = 0; j < m; ++j) ip += lut_i[(long)j*ksub + code[j]];
    Sc[(long)i*vc + v] = 1.0f - ip;
}
