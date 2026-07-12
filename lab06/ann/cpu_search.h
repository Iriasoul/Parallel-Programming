// cpu_search.h 
// S1：CPU - IVF / IVF-PQ 搜索（AVX2 + OpenMP）
// 输出统一为 CSV： method,nlist,nprobe,PQ_M,rerank_k,threads,recall,us_per_query,speedup
#pragma once
#include "cpu_index.h"

inline void run_cpu_ivf(const float* base, size_t N, const float* query, size_t M,
                        size_t d, size_t k, const int* gt, size_t gtd, int threads) {
    const int nlist = 256, PQ_M = 8;
    const double BASE = BENCH_BASELINE_US;
    std::printf("\n== [S1] CPU IVF / IVF-PQ ==  N=%zu d=%zu M=%zu k=%zu  nlist=%d PQ_M=%d threads=%d\n",
                N, d, M, k, nlist, PQ_M, threads);
    std::printf("baseline(avx2 flat) = %.1f us/query\n\n", BASE);

    CpuTimer bt; bt.start();
    build_ivf(base, N, d, nlist);
    build_ivf_pq(N, d, PQ_M);
    std::printf("build IVF+PQ: %.0f ms\n\n", bt.stop_ms());

    std::printf("method,nlist,nprobe,PQ_M,rerank_k,threads,recall,us_per_query,speedup\n");

    auto bench = [&](const char* name, int nprobe, size_t rerank_k, bool pq) {
        std::vector<std::vector<uint32_t>> res(M);
        if (pq) ivf_pq_search_omp(query, d, k, nprobe, rerank_k, threads);  // 预热
        else    ivf_search_omp(query, d, k, nprobe, threads);
        CpuTimer t; t.start();
        for (size_t i = 0; i < M; ++i)
            res[i] = pq ? ivf_pq_search_omp(query + i*d, d, k, nprobe, rerank_k, threads)
                        : ivf_search_omp   (query + i*d, d, k, nprobe, threads);
        double us = t.stop_ms() / M * 1000.0;
        double rec = compute_recall(res, gt, gtd, k);
        std::printf("%s,%d,%d,%d,%zu,%d,%.4f,%.1f,%.2f\n",
                    name, nlist, nprobe, pq ? PQ_M : 0, pq ? rerank_k : (size_t)0,
                    threads, rec, us, BASE / us);
    };

    for (int np : {8, 16, 32}) bench("IVF", np, 0, false);            // IVF 精确
    for (int np : {8, 16, 32})
        for (size_t rk : {200, 500})
            bench("IVF-PQ", np, rk, true);                            // IVF-PQ 两阶段

    std::printf("====================\n");
    free_ivf_pq(); free_ivf();
}
