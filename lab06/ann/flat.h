// flat.h 
// S0：暴力检索基线
// 对照 scalar 与 avx2，SIMD 本身的加速比
#pragma once
#include "bench_common.h"

static void flat_search(const float* base, size_t N, const float* q, size_t d,
                        size_t k, std::vector<uint32_t>& out, bool use_simd) {
    MaxHeap h;
    for (size_t j = 0; j < N; ++j) {
        float ip = use_simd ? ip_default(q, base + j * d, d)
                            : ip_scalar (q, base + j * d, d);
        heap_push(h, 1.0f - ip, (uint32_t)j, k);
    }
    out = heap_to_ids(h);
}

static double flat_run_all(const float* base, size_t N, const float* query, size_t d,
                           size_t M, size_t k, bool use_simd,
                           const int* gt, size_t gt_d, double& recall) {
    std::vector<std::vector<uint32_t>> res(M);
    CpuTimer t; t.start();
    for (size_t i = 0; i < M; ++i)
        flat_search(base, N, query + i * d, d, k, res[i], use_simd);
    double ms = t.stop_ms();
    recall = compute_recall(res, gt, gt_d, k);
    return ms;
}

inline void run_flat(const float* base, size_t N, const float* query, size_t M,
                     size_t d, size_t k, const int* gt, size_t gtd) {
    std::printf("\n== [S0] Flat暴力算法 ==  N=%zu d=%zu  测试 query=%zu  k=%zu  AVX2=%d\n",
                N, d, M, k, BENCH_HAVE_AVX2);

    double rec_s = 0, rec_v = 0;
    double ms_scalar = flat_run_all(base, N, query, d, M, k, false, gt, gtd, rec_s);
    double ms_avx2   = flat_run_all(base, N, query, d, M, k, true,  gt, gtd, rec_v);

    std::printf("\n[scalar] recall@%zu = %.4f | 总 %.1f ms | %.1f us/query\n",
                k, rec_s, ms_scalar, ms_scalar / M * 1000.0);
    std::printf("[avx2  ] recall@%zu = %.4f | 总 %.1f ms | %.1f us/query\n",
                k, rec_v, ms_avx2, ms_avx2 / M * 1000.0);
    std::printf("SIMD 加速比 (scalar/avx2) = %.2fx\n", ms_scalar / ms_avx2);
    std::printf("\n>>> 暴力基线延迟 = %.1f us/query (avx2)\n",
                ms_avx2 / M * 1000.0);
    std::printf("====================\n");
}
