// lsh_cpu.h 
// CPU - SimHash-LSH（AVX2投影 + OpenMP Hamming 粗排 + SIMD 精排）

#pragma once
#include "lsh_common.h"
#include <omp.h>

namespace lsh_ns {
    int K = 128;                        // 码长(bit)，须为 64 的倍数
    int nwords = 2;                     // K/64
    std::vector<float> planes;          // [K][d] 随机超平面
    std::vector<uint64_t> codes;        // [N][nwords] 库向量二值码
    size_t N = 0; int dim = 0;
}

// 建库：生成超平面 + 投影全库 + 打包符号位（投影 = 每条 base 与 K 个平面的内积，AVX2）
inline void lsh_build(const float* base, size_t N, int d, int K) {
    using namespace lsh_ns;
    lsh_ns::K = K; nwords = K / 64; dim = d; lsh_ns::N = N;
    lsh_gen_planes(K, d, planes);
    codes.assign(N * nwords, 0);
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)N; ++i) {
        std::vector<float> proj(K);
        const float* x = base + (size_t)i * d;
        for (int j = 0; j < K; ++j) proj[j] = ip_default(x, planes.data() + (size_t)j * d, d);
        lsh_pack_signs(proj.data(), K, codes.data() + (size_t)i * nwords);
    }
}

inline void lsh_free() { lsh_ns::planes.clear(); lsh_ns::codes.clear(); }

// 单查询：投影打包 - Hamming 粗排(OpenMP over base) 取 rerank_k - 精确内积精排取 k
inline std::vector<uint32_t>
lsh_search_omp(const float* base, const float* q, int d, size_t k, int rerank_k, int nthr) {
    using namespace lsh_ns;
    alignas(64) uint64_t qcode[64];                     // 支持到 K=4096
    { std::vector<float> proj(K);
      for (int j = 0; j < K; ++j) proj[j] = ip_default(q, planes.data() + (size_t)j * d, d);
      lsh_pack_signs(proj.data(), K, qcode); }

    std::vector<MaxHeap> local(nthr);                   // 线程内局部 Hamming 粗排
    #pragma omp parallel num_threads(nthr)
    {
        int tid = omp_get_thread_num();
        MaxHeap& h = local[tid];
        #pragma omp for schedule(static) nowait
        for (long i = 0; i < (long)N; ++i) {
            int ham = lsh_hamming(qcode, codes.data() + (size_t)i * nwords, nwords);
            heap_push(h, (float)ham, (uint32_t)i, (size_t)rerank_k);
        }
    }
    MaxHeap coarse;                                     // 归并成全局 rerank_k 候选
    for (auto& h : local)
        while (!h.empty()) { auto t = h.top(); h.pop(); heap_push(coarse, t.first, t.second, (size_t)rerank_k); }

    MaxHeap fine;                                       // 候选精排
    while (!coarse.empty()) {
        uint32_t id = coarse.top().second; coarse.pop();
        heap_push(fine, 1.0f - ip_default(q, base + (size_t)id * d, d), id, k);
    }
    return heap_to_ids(fine);
}

// S5 入口：扫 K × rerank_k 的 Recall-Latency
inline void run_lsh_cpu(const float* base, size_t N, const float* query, size_t M,
                        size_t d, size_t k, const int* gt, size_t gtd, int threads) {
    const double BASE = BENCH_BASELINE_US;
    std::printf("\n== [S3] LSH - CPU ==  N=%zu d=%zu M=%zu k=%zu threads=%d\n", N, d, M, k, threads);
    std::printf("baseline(avx2 flat) = %.1f us/query\n\n", BASE);
    std::printf("method,K_bits,rerank_k,threads,recall,us_per_query,speedup\n");

    for (int K : {64, 128, 256}) {
        CpuTimer bt; bt.start(); lsh_build(base, N, (int)d, K);
        double build_ms = bt.stop_ms();
        for (int rk : {200, 500, 1000}) {
            std::vector<std::vector<uint32_t>> res(M);
            lsh_search_omp(base, query, (int)d, k, rk, threads);        // 预热
            CpuTimer t; t.start();
            for (size_t i = 0; i < M; ++i)
                res[i] = lsh_search_omp(base, query + i*d, (int)d, k, rk, threads);
            double us = t.stop_ms() / M * 1000.0;
            double rec = compute_recall(res, gt, gtd, k);
            std::printf("LSH,%d,%d,%d,%.4f,%.1f,%.2f   (build %.0f ms)\n",
                        K, rk, threads, rec, us, BASE / us, build_ms);
        }
        lsh_free();
    }
    std::printf("====================\n");
}
