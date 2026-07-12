// gpu_coarse.cuh 
// S2：GPU 粗排（cuBLAS GEMM + 设备侧 Top-k）
// 每批 batch_B 条 query 一次 GEMM，显存 O(batch_B * N)
#pragma once
#include "bench_common.h"
#include "gpu_util.cuh"

inline void run_gpu_coarse(const float* base, size_t N_, const float* query, size_t M,
                           size_t d_, size_t k, const int* gt, size_t gtd, int batch_B) {
    const int N = (int)N_, d = (int)d_;
    const double BASE = BENCH_BASELINE_US;
    std::printf("\n== [S2] GPU cuBLAS ==\n");
    std::printf("N=%d d=%d M=%zu k=%zu batch_B=%d   baseline=%.1f us/query\n\n", N, d, M, k, batch_B, BASE);

    cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h));
    float *dB, *dQ, *dS; uint32_t* dIdx; float* dDist;
    CUDA_CHECK(cudaMalloc(&dB,  (size_t)N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dQ,  (size_t)batch_B*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dS,  (size_t)batch_B*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dIdx,(size_t)batch_B*k*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dDist,(size_t)batch_B*k*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dB, base, (size_t)N*d*sizeof(float), cudaMemcpyHostToDevice));

    std::vector<std::vector<uint32_t>> res(M);
    const float alpha = 1.f, beta = 0.f;
    int nt = 256; size_t sh = (size_t)nt * (sizeof(float) + sizeof(int));

    auto run_once = [&](bool timed)->float {
        GpuTimer gt; if (timed) gt.start();
        for (size_t off = 0; off < M; off += batch_B) {
            int B = (int)std::min<size_t>(batch_B, M - off);
            CUDA_CHECK(cudaMemcpy(dQ, query + off*d, (size_t)B*d*sizeof(float), cudaMemcpyHostToDevice));
            // dS(N×B col-major = [B][N] row-major) = B^T·Q
            CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, B, d,
                                     &alpha, dB, d, dQ, d, &beta, dS, N));
            size_t total = (size_t)B*N;
            ip_to_dist<<<(int)((total+255)/256), 256>>>(dS, total);
            topk_rows<<<B, nt, sh>>>(dS, N, (int)k, dIdx, dDist);
            CUDA_CHECK(cudaGetLastError());
            std::vector<uint32_t> hIdx((size_t)B*k);
            CUDA_CHECK(cudaMemcpy(hIdx.data(), dIdx, (size_t)B*k*sizeof(uint32_t), cudaMemcpyDeviceToHost));
            if (!timed) for (int i=0;i<B;++i) res[off+i].assign(hIdx.begin()+(size_t)i*k, hIdx.begin()+(size_t)(i+1)*k);
        }
        return timed ? gt.stop_ms() : 0.f;
    };

    run_once(false);                                   // 正确性
    double rec = compute_recall(res, gt, gtd, k);
    run_once(true);                                    // 预热
    double ms = 0; const int R = 3; for (int r=0;r<R;++r) ms += run_once(true); ms /= R;

    std::printf("method,N,d,batch_B,recall,us_per_query,speedup\n");
    std::printf("GPU-flat-exact,%d,%d,%d,%.4f,%.1f,%.2f\n",
                N, d, batch_B, rec, ms/M*1000.0, BASE/(ms/M*1000.0));
    std::printf("====================\n");

    cudaFree(dB); cudaFree(dQ); cudaFree(dS); cudaFree(dIdx); cudaFree(dDist); cublasDestroy(h);
}
