// lsh_gpu.cuh 
// GPU -  SimHash-LSH（cuBLAS 投影 + popcount Hamming + 设备精排）

#pragma once
#include "bench_common.h"
#include "lsh_common.h"
#include "gpu_util.cuh"

#define LSH_MAXC 4096

// GPU kernels
__global__ void lsh_pack_kernel(const float* proj, int rows, int K, unsigned long long* codes) {
    int nwords = K / 64;
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= (long)rows * nwords) return;
    int row = idx / nwords, w = idx % nwords;
    const float* pr = proj + (size_t)row * K + w * 64;
    unsigned long long word = 0;
    for (int b = 0; b < 64; ++b) if (pr[b] >= 0.f) word |= (1ull << b);
    codes[(size_t)row * nwords + w] = word;
}

__global__ void lsh_hamming_kernel(const unsigned long long* qcodes, int B,
                                   const unsigned long long* codes, int N, int nwords, int* dHam) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long tot = (long)B * N;
    if (idx >= tot) return;
    int b = idx / N, i = idx % N;
    const unsigned long long* qc = qcodes + (size_t)b * nwords;
    const unsigned long long* cc = codes  + (size_t)i * nwords;
    int h = 0;
    for (int w = 0; w < nwords; ++w) h += __popcll(qc[w] ^ cc[w]);
    dHam[idx] = h;
}

__global__ void lsh_query_kernel(const float* dQ, int B, const float* dBase, int N, int d,
                                 const int* dHam, int K, int rerank_k, int k, uint32_t* out_idx) {
    int b = blockIdx.x; if (b >= B) return;
    int tid = threadIdx.x, nt = blockDim.x;
    extern __shared__ char smem[];
    int*   hist  = (int*)smem;
    int*   cand  = (int*)(hist + (K + 1));
    float* cdist = (float*)(cand + LSH_MAXC);
    float* qv    = (float*)(cdist + LSH_MAXC);
    __shared__ int radius; __shared__ int ccount;

    for (int j = tid; j < d;   j += nt) qv[j]   = dQ[(size_t)b * d + j];
    for (int h = tid; h <= K;  h += nt) hist[h] = 0;
    if (tid == 0) { radius = 0; ccount = 0; }
    __syncthreads();

    const int* ham = dHam + (size_t)b * N;
    for (int i = tid; i < N; i += nt) atomicAdd(&hist[ham[i]], 1);
    __syncthreads();

    if (tid == 0) {
        int cum = 0, r = K;
        for (int h = 0; h <= K; ++h) { cum += hist[h]; if (cum >= rerank_k) { r = h; break; } }
        radius = r;
    }
    __syncthreads();

    for (int i = tid; i < N; i += nt)
        if (ham[i] <= radius) { int p = atomicAdd(&ccount, 1); if (p < LSH_MAXC) cand[p] = i; }
    __syncthreads();

    int c = ccount; if (c > LSH_MAXC) c = LSH_MAXC;
    for (int p = tid; p < c; p += nt) {
        const float* bv = dBase + (size_t)cand[p] * d;
        float ip = 0.f; for (int j = 0; j < d; ++j) ip += qv[j] * bv[j];
        cdist[p] = 1.f - ip;
    }
    __syncthreads();

    if (tid == 0) {
        for (int it = 0; it < k; ++it) {
            float best = FLT_MAX; int bi = -1;
            for (int p = 0; p < c; ++p) if (cdist[p] < best) { best = cdist[p]; bi = p; }
            if (bi >= 0) { out_idx[(size_t)b * k + it] = (uint32_t)cand[bi]; cdist[bi] = FLT_MAX; }
            else          out_idx[(size_t)b * k + it] = 0;
        }
    }
}

// lsh复用
namespace lsh_gpu_ns {
    int Kbits = 256, nwords = 4, Nbase = 0, dim = 0, max_batch = 512;
    cublasHandle_t handle = nullptr;
    float *dBase = nullptr, *dPlanes = nullptr, *dProjQ = nullptr, *dQ = nullptr;
    unsigned long long *dCodes = nullptr, *dQcodes = nullptr;
    int* dHam = nullptr; uint32_t* dIdx = nullptr;
    bool built = false;
}

// 建 GPU LSH 索引
inline void lsh_gpu_build(const float* base, int N, int d, int K, int batch_B, cublasHandle_t h) {
    using namespace lsh_gpu_ns;
    Kbits = K; nwords = K / 64; Nbase = N; dim = d; max_batch = batch_B; handle = h;
    CUDA_CHECK(cudaMalloc(&dBase,   (size_t)N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dPlanes, (size_t)K*d*sizeof(float)));
    // 查询暂存
    CUDA_CHECK(cudaMalloc(&dProjQ,  (size_t)max_batch*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dQ,      (size_t)max_batch*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dQcodes, (size_t)max_batch*nwords*sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&dHam,    (size_t)max_batch*N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dIdx,    (size_t)max_batch*10*sizeof(uint32_t)));   // k 固定 10

    CUDA_CHECK(cudaMemcpy(dBase, base, (size_t)N*d*sizeof(float), cudaMemcpyHostToDevice));
    // 生成超平面
    std::vector<float> planes; lsh_gen_planes(K, d, planes);
    CUDA_CHECK(cudaMemcpy(dPlanes, planes.data(), (size_t)K*d*sizeof(float), cudaMemcpyHostToDevice));
    // 建库投影 GEMM + 打包
    float* dProjBase; CUDA_CHECK(cudaMalloc(&dProjBase, (size_t)N*K*sizeof(float)));
    CUBLAS_CHECK(cublasSetStream(h, 0));
    const float alpha=1.f, beta=0.f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, K, N, d, &alpha, dPlanes, d, dBase, d, &beta, dProjBase, K));
    CUDA_CHECK(cudaMalloc(&dCodes, (size_t)N*nwords*sizeof(unsigned long long)));
    lsh_pack_kernel<<<(int)(((size_t)N*nwords + 255)/256), 256>>>(dProjBase, N, K, dCodes);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dProjBase);
    built = true;
}

// 用 GPU LSH 检索 query 区间 [lo,hi)，结果写入 out[i]
// 要先 lsh_gpu_build
inline void lsh_gpu_search(const float* query, size_t k, int rerank_k,
                           size_t lo, size_t hi,
                           std::vector<std::vector<uint32_t>>& out) {
    using namespace lsh_gpu_ns;
    if (!built || hi <= lo) return;
    size_t cnt = hi - lo;
    size_t sh_q = (size_t)(Kbits+1)*sizeof(int) + (size_t)LSH_MAXC*(sizeof(int)+sizeof(float)) + (size_t)dim*sizeof(float);
    const float alpha=1.f, beta=0.f;

    for (size_t off = 0; off < cnt; off += max_batch) {
        int B = (int)std::min<size_t>(max_batch, cnt - off);
        size_t q_lo = lo + off;
        CUDA_CHECK(cudaMemcpy(dQ, query + q_lo*dim, (size_t)B*dim*sizeof(float), cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasSetStream(handle, 0));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Kbits, B, dim,
                                 &alpha, dPlanes, dim, dQ, dim, &beta, dProjQ, Kbits));
        lsh_pack_kernel<<<(int)(((size_t)B*nwords + 255)/256), 256>>>(dProjQ, B, Kbits, dQcodes);
        lsh_hamming_kernel<<<(int)(((size_t)B*Nbase + 255)/256), 256>>>(dQcodes, B, dCodes, Nbase, nwords, dHam);
        lsh_query_kernel<<<B, 256, sh_q>>>(dQ, B, dBase, Nbase, dim, dHam, Kbits, rerank_k, (int)k, dIdx);
        CUDA_CHECK(cudaGetLastError());
        std::vector<uint32_t> hIdx((size_t)B*k);
        CUDA_CHECK(cudaMemcpy(hIdx.data(), dIdx, (size_t)B*k*sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for (int i = 0; i < B; ++i)
            out[q_lo + i].assign(hIdx.begin() + (size_t)i*k, hIdx.begin() + (size_t)(i + 1)*k);
    }
}

inline void lsh_gpu_free() {
    using namespace lsh_gpu_ns;
    cudaFree(dBase); cudaFree(dPlanes); cudaFree(dProjQ); cudaFree(dQ);
    cudaFree(dQcodes); cudaFree(dHam); cudaFree(dIdx); cudaFree(dCodes);
    dBase = dPlanes = dProjQ = dQ = nullptr; dQcodes = nullptr;
    dHam = nullptr; dIdx = nullptr; dCodes = nullptr;
    built = false;
}

// 入口
inline void run_lsh_gpu(const float* base, size_t N_, const float* query, size_t M,
                        size_t d_, size_t k, const int* gt, size_t gtd, int batch_B) {
    const int N = (int)N_, d = (int)d_;
    const double BASE = BENCH_BASELINE_US;
    std::printf("\n== [S4] LSH - GPU ==\n");
    std::printf("N=%d d=%d M=%zu k=%zu batch_B=%d  baseline=%.1f us/q\n\n", N, d, M, k, batch_B, BASE);

    cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h));
    std::printf("method,K_bits,rerank_k,recall,us_per_query,speedup\n");
    for (int K : {64, 128, 256}) {
        lsh_gpu_build(base, N, d, K, batch_B, h);
        for (int rk : {200, 500, 1000}) {
            std::vector<std::vector<uint32_t>> res(M);
            lsh_gpu_search(query, k, rk, 0, M, res);         // 预热
            GpuTimer gtm; gtm.start();
            for (size_t i = 0; i < M; ++i) res[i].clear();
            lsh_gpu_search(query, k, rk, 0, M, res);
            float ms = gtm.stop_ms();
            double rec = compute_recall(res, gt, gtd, k);
            double us = ms / M * 1000.0;
            std::printf("LSH-GPU,%d,%d,%.4f,%.1f,%.2f\n", K, rk, rec, us, BASE / us);
        }
        lsh_gpu_free();
    }
    std::printf("====================\n");
    cublasDestroy(h);
}
