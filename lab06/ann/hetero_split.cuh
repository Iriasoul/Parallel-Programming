// hetero_split.cuh
// S5：融合算法-异构协同分流（GPU-LSH ∥ CPU-IVF）

// GPU 上 LSH 最优；CPU 上 IVF 最优
// 两块硬件并发处理不相交的查询段，吞吐 ≈ qps_LSH + qps_IVF

#pragma once
#include "cpu_index.h"      // build_ivf / cpu_ivf_search_range / free_ivf
#include "lsh_gpu.cuh"      // lsh_gpu_build / lsh_gpu_search / lsh_gpu_free
#include "gpu_util.cuh"
#include <thread>

// GPU 精确检索 query[lo,hi)，结果写 out（绝对下标）
static void gpu_exact_search_range(const float* base, int N, const float* query, int d,
                                   size_t k, size_t lo, size_t hi, int batch_B,
                                   cublasHandle_t h, float* dB,
                                   std::vector<std::vector<uint32_t>>& out) {
    if (hi <= lo) return; size_t cnt = hi - lo;
    float *dQ, *dS; uint32_t* dIdx;
    CUDA_CHECK(cudaMalloc(&dQ,  (size_t)batch_B*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dS,  (size_t)batch_B*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dIdx,(size_t)batch_B*k*sizeof(uint32_t)));
    const float alpha=1.f, beta=0.f; int nt=256; size_t sh=(size_t)nt*(sizeof(float)+sizeof(int));
    CUBLAS_CHECK(cublasSetStream(h, 0));
    for (size_t off = 0; off < cnt; off += batch_B) {
        int B = (int)std::min<size_t>(batch_B, cnt - off);
        CUDA_CHECK(cudaMemcpy(dQ, query+(lo+off)*d, (size_t)B*d*sizeof(float), cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, B, d, &alpha, dB, d, dQ, d, &beta, dS, N));
        size_t tot=(size_t)B*N; ip_to_dist<<<(int)((tot+255)/256),256>>>(dS, tot);
        topk_rows<<<B, nt, sh>>>(dS, N, (int)k, dIdx, nullptr);
        std::vector<uint32_t> hIdx((size_t)B*k);
        CUDA_CHECK(cudaMemcpy(hIdx.data(), dIdx, (size_t)B*k*sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for (int i=0;i<B;++i) out[lo+off+i].assign(hIdx.begin()+(size_t)i*k, hIdx.begin()+(size_t)(i+1)*k);
    }
    cudaFree(dQ); cudaFree(dS); cudaFree(dIdx);
}

// S5 入口
// gpu_algo=0 : GPU-exact (baseline) ; gpu_algo=1 : GPU-LSH (default)
// LSH 参数：K_lsh=256, rerank_lsh 固定 500；可调
inline void run_hetero_split(const float* base, size_t N_, const float* query, size_t M,
                             size_t d_, size_t k, const int* gt, size_t gtd,
                             int batch_B, int nthr, int nprobe, int gpu_algo, int K_lsh, int rerank_lsh) {
    const int N = (int)N_, d = (int)d_, nlist = 256;
    const double BASE = BENCH_BASELINE_US;
    const char* gpu_name = (gpu_algo == 0) ? "exact" : "LSH";
    std::printf("\n== [S5] 异构协同分流 (GPU-%s || CPU-IVF) ==\n", gpu_name);
    std::printf("N=%d d=%d M=%zu k=%zu batch_B=%d CPU=%d线程 nprobe=%d  baseline=%.1f us/q\n",
                N, d, M, k, batch_B, nthr, nprobe, BASE);
    if (gpu_algo == 1)
        std::printf("GPU-LSH: K=%d rerank_k=%d\n", K_lsh, rerank_lsh);

    // 准备：GPU 常驻资源 + CPU IVF 索引 + LSH
    cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h));
    float* dB; CUDA_CHECK(cudaMalloc(&dB, (size_t)N*d*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dB, base, (size_t)N*d*sizeof(float), cudaMemcpyHostToDevice));
    CpuTimer bt; bt.start(); build_ivf(base, N, d, nlist);
    if (gpu_algo == 1) lsh_gpu_build(base, N, d, K_lsh, batch_B, h);
    std::printf("build indices: %.0f ms\n", bt.stop_ms());

    // 锁页 query，保证 H2D 异步，与 CPU 段并行不串行化
    CUDA_CHECK(cudaHostRegister((void*)query, (size_t)M*d*sizeof(float), cudaHostRegisterDefault));
    cudaStream_t st=0;
    // 异步入队 GPU 段
    // exact 用 stream 0 + cudaMemcpyAsync；LSH 因核内调用 cublasSgemm 设流，这里包装成串行
    // 为保持并行，GPU 段在专用 CPU 线程里同步执行，主线程同时跑 CPU IVF
    std::vector<std::vector<uint32_t>> res(M);

    auto run = [&](double f, bool fill)->double {
        size_t g = (size_t)(f*M + 0.5); if (g > M) g = M;
        for (auto& r : res) r.clear();

        auto gpu_job = [&]() {
            if (gpu_algo == 0)
                gpu_exact_search_range(base, N, query, d, k, 0, g, batch_B, h, dB, res);
            else
                lsh_gpu_search(query, k, rerank_lsh, 0, g, res);
        };

        CpuTimer t; t.start();
        // GPU 段在独立线程跑（与主线程的 CPU IVF 并发）
        std::thread gpu_thr(gpu_job);
        if (g < M)
            cpu_ivf_search_range(query, d, k, nprobe, g, M, res, nthr);
        gpu_thr.join();
        // 如果没走填充（fill=false 时为计时），GPU 结果也要写出
        if (fill && g > 0 && res[0].empty()) gpu_job();
        return t.stop_ms();
    };

    auto timed = [&](double f)->std::pair<double,double> {
        run(f, true);  double rec = compute_recall(res, gt, gtd, k);
        run(f, false); double s = 0; const int R = 3; for (int r=0;r<R;++r) s += run(f, false); s /= R;
        return {rec, s};
    };

    std::printf("\ngpu_frac,recall,us_per_query,qps,speedup_vs_baseline\n");
    for (double f : {1.0, 0.96, 0.94, 0.92, 0.9, 0.8, 0.6, 0.4, 0.0}) {
        auto pr = timed(f); double ms = pr.second, rec = pr.first;
        std::printf("%.2f,%.4f,%.1f,%.0f,%.2f\n", f, rec, ms/M*1000.0, M/(ms/1000.0), BASE/(ms/M*1000.0));
    }
    std::printf("====================\n");

    cudaHostUnregister((void*)query); cudaFree(dB); cublasDestroy(h);
    free_ivf(); if (gpu_algo == 1) lsh_gpu_free();
}
