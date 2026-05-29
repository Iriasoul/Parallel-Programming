#pragma once

#include <queue>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <omp.h>
#include <pthread.h>
#include <arm_neon.h>
#include "thread_pool.h"

namespace pq_ns {
    const int PQ_M = 8;                  // 子空间数
    const int PQ_K = 256;                // 每个子空间的簇数 (uint8 编码)
    uint8_t* pq_codes = nullptr;         // AoS: [N][M]
    uint8_t* pq_codes_soa = nullptr;     // SoA: [M][N], cache 友好
    float*   pq_centroids = nullptr;     // [M][K][sub_dim]
    float*   pq_centroids_soa = nullptr; // [M][sub_dim][K], LUT 构建用
    float*   pq_dists_buf = nullptr;     // 预分配 ADC 累加缓冲区 [N], 避免 per-query malloc
}

// SIMD 内积
static inline float pq_neon_ip(const float* a, const float* b, size_t dim) {
    float32x4_t s = vdupq_n_f32(0.0f);
    size_t d = 0;
    for (; d + 3 < dim; d += 4) {
        s = vfmaq_f32(s, vld1q_f32(a + d), vld1q_f32(b + d));
    }
    float sum = vaddvq_f32(s);
    for (; d < dim; ++d) sum += a[d] * b[d];
    return sum;
}

// 简化 KMeans
inline void pq_kmeans(float* data, size_t n, size_t dim, float* cents, int iters = 30) {
    using namespace pq_ns;
    for (int i = 0; i < PQ_K; ++i) {
        std::memcpy(cents + i * dim, data + (rand() % n) * dim, dim * sizeof(float));
    }
    for (int it = 0; it < iters; ++it) {
        std::vector<float> next(PQ_K * dim, 0);
        std::vector<int> cnt(PQ_K, 0);
        for (size_t i = 0; i < n; ++i) {
            float best_ip = -1e30f; int best = 0;
            for (int c = 0; c < PQ_K; ++c) {
                float ip = pq_neon_ip(data + i * dim, cents + c * dim, dim);
                if (ip > best_ip) { best_ip = ip; best = c; }
            }
            cnt[best]++;
            for (size_t d = 0; d < dim; ++d) next[best * dim + d] += data[i * dim + d];
        }
        for (int c = 0; c < PQ_K; ++c) {
            if (cnt[c]) for (size_t d = 0; d < dim; ++d) cents[c * dim + d] = next[c * dim + d] / cnt[c];
        }
    }
}

// 离线: PQ 训练 + 编码 + SoA 布局准备
inline void pq_offline(float* base, size_t base_number, size_t vecdim) {
    using namespace pq_ns;
    size_t sub_dim = vecdim / PQ_M;
    pq_codes = new uint8_t[base_number * PQ_M];
    pq_centroids = new float[PQ_M * PQ_K * sub_dim];

    size_t train_n = std::min((size_t)20000, base_number);
    for (int m = 0; m < PQ_M; ++m) {
        std::vector<float> sub(train_n * sub_dim);
        for (size_t i = 0; i < train_n; ++i) {
            std::memcpy(&sub[i * sub_dim],
                        base + i * vecdim + m * sub_dim,
                        sub_dim * sizeof(float));
        }
        pq_kmeans(sub.data(), train_n, sub_dim, pq_centroids + m * PQ_K * sub_dim, 30);
    }

    // 全库编码 (按 base 向量并行)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < base_number; ++i) {
        for (int m = 0; m < PQ_M; ++m) {
            float* sub_v = base + i * vecdim + m * sub_dim;
            float* cents = pq_centroids + m * PQ_K * sub_dim;
            float best_ip = -1e30f; int best = 0;
            for (int c = 0; c < PQ_K; ++c) {
                float ip = pq_neon_ip(sub_v, cents + c * sub_dim, sub_dim);
                if (ip > best_ip) { best_ip = ip; best = c; }
            }
            pq_codes[i * PQ_M + m] = (uint8_t)best;
        }
    }

    // SoA codes: [M][N]
    pq_codes_soa = new uint8_t[PQ_M * base_number];
    for (int m = 0; m < PQ_M; ++m) {
        for (size_t i = 0; i < base_number; ++i) {
            pq_codes_soa[m * base_number + i] = pq_codes[i * PQ_M + m];
        }
    }
    // SoA centroids: [M][sub_dim][K]
    pq_centroids_soa = new float[PQ_M * sub_dim * PQ_K];
    for (int m = 0; m < PQ_M; ++m) {
        for (int d = 0; d < (int)sub_dim; ++d) {
            for (int c = 0; c < PQ_K; ++c) {
                pq_centroids_soa[m * sub_dim * PQ_K + d * PQ_K + c] =
                    pq_centroids[m * PQ_K * sub_dim + c * sub_dim + d];
            }
        }
    }

    // 预分配 ADC 累加缓冲区, 避免 pq_search_final_baseline 每次 query 都 malloc+memset
    pq_dists_buf = new float[base_number];
}

inline void pq_free() {
    using namespace pq_ns;
    if (pq_codes) { delete[] pq_codes; pq_codes = nullptr; }
    if (pq_codes_soa) { delete[] pq_codes_soa; pq_codes_soa = nullptr; }
    if (pq_centroids) { delete[] pq_centroids; pq_centroids = nullptr; }
    if (pq_centroids_soa) { delete[] pq_centroids_soa; pq_centroids_soa = nullptr; }
    if (pq_dists_buf) { delete[] pq_dists_buf; pq_dists_buf = nullptr; }
}

// S3 PQ-Rerank-SIMD
inline std::priority_queue<std::pair<float, uint32_t>>
pq_search_baseline(float* query, float* base,
                   size_t base_number, size_t vecdim,
                   size_t k, size_t rerank_k)
{
    using namespace pq_ns;
    size_t sub_dim = vecdim / PQ_M;
    float lut[PQ_M][PQ_K];

    // LUT 构建 (单线程, 串行扫描所有子空间 + centroids)
    for (int m = 0; m < PQ_M; ++m) {
        float* sub_q = query + m * sub_dim;
        float* m_cent = pq_centroids + m * PQ_K * sub_dim;
        for (int c = 0; c < PQ_K; ++c) {
            lut[m][c] = pq_neon_ip(sub_q, m_cent + c * sub_dim, sub_dim);
        }
    }

    // ADC 粗排: 单线程扫描全库, 维护 rerank_k 大小的优先队列
    std::priority_queue<std::pair<float, uint32_t>> coarse_q;
    for (size_t i = 0; i < base_number; ++i) {
        float ip = 0.0f;
        uint8_t* code = pq_codes + i * PQ_M;  // AoS 布局
        for (int m = 0; m < PQ_M; ++m) ip += lut[m][code[m]];
        float dis = 1.0f - ip;
        if (coarse_q.size() < rerank_k || dis < coarse_q.top().first) {
            coarse_q.push({dis, (uint32_t)i});
            if (coarse_q.size() > rerank_k) coarse_q.pop();
        }
    }

    // 精排
    std::priority_queue<std::pair<float, uint32_t>> fine_q;
    while (!coarse_q.empty()) {
        uint32_t idx = coarse_q.top().second;
        coarse_q.pop();
        float exact_dis = 1.0f - pq_neon_ip(query, base + idx * vecdim, vecdim);
        if (fine_q.size() < k || exact_dis < fine_q.top().first) {
            fine_q.push({exact_dis, idx});
            if (fine_q.size() > k) fine_q.pop();
        }
    }
    return fine_q;
}

// S4: PQ-Final SoA 跨 centroid SIMD
inline std::priority_queue<std::pair<float, uint32_t>>
pq_search_final_baseline(float* query, float* base,
                         size_t base_number, size_t vecdim,
                         size_t k, size_t rerank_k)
{
    using namespace pq_ns;
    size_t sub_dim = vecdim / PQ_M;
    alignas(64) float lut[PQ_M][PQ_K];

    // LUT 构建 (跨 centroid SIMD, 单线程)
    for (int m = 0; m < PQ_M; ++m) {
        float* sub_q = query + m * sub_dim;
        float* m_soa = pq_centroids_soa + m * sub_dim * PQ_K;
        for (int c = 0; c < PQ_K; c += 4) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (size_t d = 0; d < sub_dim; ++d) {
                acc = vfmaq_f32(acc, vdupq_n_f32(sub_q[d]),
                                vld1q_f32(m_soa + d * PQ_K + c));
            }
            vst1q_f32(&lut[m][c], acc);
        }
    }

    // ADC 粗排
    float* dists = pq_dists_buf;
    std::fill(dists, dists + base_number, 0.0f);
    for (int m = 0; m < PQ_M; ++m) {
        const uint8_t* col = pq_codes_soa + (size_t)m * base_number;
        for (size_t i = 0; i < base_number; ++i)
            dists[i] += lut[m][col[i]];
    }

    std::priority_queue<std::pair<float, uint32_t>> coarse_q;
    for (size_t i = 0; i < base_number; ++i) {
        float dis = 1.0f - dists[i];
        if (coarse_q.size() < rerank_k || dis < coarse_q.top().first) {
            coarse_q.push({dis, (uint32_t)i});
            if (coarse_q.size() > rerank_k) coarse_q.pop();
        }
    }

    // 精排 (单线程)
    std::priority_queue<std::pair<float, uint32_t>> fine_q;
    while (!coarse_q.empty()) {
        uint32_t idx = coarse_q.top().second;
        coarse_q.pop();
        float exact_dis = 1.0f - pq_neon_ip(query, base + idx * vecdim, vecdim);
        if (fine_q.size() < k || exact_dis < fine_q.top().first) {
            fine_q.push({exact_dis, idx});
            if (fine_q.size() > k) fine_q.pop();
        }
    }
    return fine_q;
}

// S5: PQ-SIMD + OpenMP
inline std::priority_queue<std::pair<float, uint32_t>>
pq_search_thread(float* query, float* base,
                 size_t base_number, size_t vecdim,
                 size_t k, size_t rerank_k,
                 int num_threads = 0)
{
    using namespace pq_ns;
    size_t sub_dim = vecdim / PQ_M;
    alignas(64) float lut[PQ_M][PQ_K];

    int nthr = (num_threads > 0) ? num_threads : omp_get_max_threads();

    // LUT 构建 (子空间外并行 + 跨 centroid SIMD 内并行)
    #pragma omp parallel for schedule(static) num_threads(nthr)
    for (int m = 0; m < PQ_M; ++m) {
        float* sub_q = query + m * sub_dim;
        float* m_soa = pq_centroids_soa + m * sub_dim * PQ_K;
        for (int c = 0; c < PQ_K; c += 4) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (size_t d = 0; d < sub_dim; ++d) {
                acc = vfmaq_f32(acc, vdupq_n_f32(sub_q[d]),
                                vld1q_f32(m_soa + d * PQ_K + c));
            }
            vst1q_f32(&lut[m][c], acc);
        }
    }

    // ADC 粗排 + 线程内精排 (流水线) 
    std::vector<std::vector<std::pair<float, uint32_t>>> thread_results(nthr);

    #pragma omp parallel num_threads(nthr)
    {
        int tid = omp_get_thread_num();
        // 手动计算 static 分段 (等价于 schedule(static)), 以便做 loop interchange
        size_t chunk = (base_number + (size_t)nthr - 1) / (size_t)nthr;
        size_t start = (size_t)tid * chunk;
        size_t end   = std::min(base_number, start + chunk);
        size_t len   = end - start;

        // Loop interchange: 外层 m (每次访问连续 len 字节), 内层 i (顺序)
        std::vector<float> local_d(len, 0.0f);
        for (int m = 0; m < PQ_M; ++m) {
            const uint8_t* col = pq_codes_soa + (size_t)m * base_number + start;
            for (size_t j = 0; j < len; ++j)
                local_d[j] += lut[m][col[j]];
        }

        std::priority_queue<std::pair<float, uint32_t>> local_coarse;
        for (size_t j = 0; j < len; ++j) {
            float dis = 1.0f - local_d[j];
            if (local_coarse.size() < rerank_k || dis < local_coarse.top().first) {
                local_coarse.push({dis, (uint32_t)(start + j)});
                if (local_coarse.size() > rerank_k) local_coarse.pop();
            }
        }

        // 线程内精排
        std::priority_queue<std::pair<float, uint32_t>> local_fine;
        while (!local_coarse.empty()) {
            uint32_t idx = local_coarse.top().second;
            local_coarse.pop();
            float exact_ip = pq_neon_ip(query, base + idx * vecdim, vecdim);
            float exact_dis = 1.0f - exact_ip;
            if (local_fine.size() < k || exact_dis < local_fine.top().first) {
                local_fine.push({exact_dis, idx});
                if (local_fine.size() > k) local_fine.pop();
            }
        }

        thread_results[tid].reserve(k);
        while (!local_fine.empty()) {
            thread_results[tid].push_back(local_fine.top());
            local_fine.pop();
        }
    }

    // 全局归并
    std::priority_queue<std::pair<float, uint32_t>> final_q;
    for (const auto& tr : thread_results) {
        for (const auto& r : tr) {
            if (final_q.size() < k || r.first < final_q.top().first) {
                final_q.push(r);
                if (final_q.size() > k) final_q.pop();
            }
        }
    }
    return final_q;
}
// S6: Pthread 版本 
struct PQPthreadArgs {
    int tid;
    int num_threads;
    float* query;
    float* base;
    size_t base_number;
    size_t vecdim;
    size_t k;
    size_t rerank_k;
    float (*lut)[256];                                // 共享 LUT
    pthread_barrier_t* barrier;                       // LUT 阶段同步
    std::vector<std::pair<float, uint32_t>>* result;  // 输出 (局部 top-k)
};

inline void* pq_pthread_worker(void* arg) {
    PQPthreadArgs* a = (PQPthreadArgs*)arg;
    using namespace pq_ns;
    size_t sub_dim = a->vecdim / PQ_M;

    // LUT 构建 (tid==0 串行做)
    if (a->tid == 0) {
        for (int m = 0; m < PQ_M; ++m) {
            float* sub_q = a->query + m * sub_dim;
            float* m_soa = pq_centroids_soa + m * sub_dim * PQ_K;
            for (int c = 0; c < PQ_K; c += 4) {
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (size_t d = 0; d < sub_dim; ++d) {
                    acc = vfmaq_f32(acc, vdupq_n_f32(sub_q[d]),
                                    vld1q_f32(m_soa + d * PQ_K + c));
                }
                vst1q_f32(&a->lut[m][c], acc);
            }
        }
    }
    // 显式 barrier: 等 LUT 全部完成, 其他线程在这里阻塞
    pthread_barrier_wait(a->barrier);

    // ADC 粗排 
    size_t chunk = (a->base_number + (size_t)a->num_threads - 1) / (size_t)a->num_threads;
    size_t start = (size_t)a->tid * chunk;
    size_t end   = std::min(a->base_number, start + chunk);
    size_t len   = end - start;

    std::vector<float> local_d(len, 0.0f);
    for (int m = 0; m < PQ_M; ++m) {
        const uint8_t* col = pq_codes_soa + (size_t)m * a->base_number + start;
        for (size_t j = 0; j < len; ++j)
            local_d[j] += a->lut[m][col[j]];
    }

    std::priority_queue<std::pair<float, uint32_t>> local_coarse;
    for (size_t j = 0; j < len; ++j) {
        float dis = 1.0f - local_d[j];
        if (local_coarse.size() < a->rerank_k || dis < local_coarse.top().first) {
            local_coarse.push({dis, (uint32_t)(start + j)});
            if (local_coarse.size() > a->rerank_k) local_coarse.pop();
        }
    }

    // 线程独立精排自己的局部候选
    std::priority_queue<std::pair<float, uint32_t>> local_fine;
    while (!local_coarse.empty()) {
        uint32_t idx = local_coarse.top().second;
        local_coarse.pop();
        float exact_dis = 1.0f - pq_neon_ip(a->query, a->base + idx * a->vecdim, a->vecdim);
        if (local_fine.size() < a->k || exact_dis < local_fine.top().first) {
            local_fine.push({exact_dis, idx});
            if (local_fine.size() > a->k) local_fine.pop();
        }
    }

    a->result->reserve(a->k);
    while (!local_fine.empty()) {
        a->result->push_back(local_fine.top());
        local_fine.pop();
    }
    return nullptr;
}

inline std::priority_queue<std::pair<float, uint32_t>>
pq_search_pthread(float* query, float* base,
                  size_t base_number, size_t vecdim,
                  size_t k, size_t rerank_k,
                  int num_threads = 4, ThreadPool* pool = nullptr)
{
    alignas(64) float lut[pq_ns::PQ_M][256];
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, nullptr, num_threads);

    std::vector<PQPthreadArgs> args(num_threads);
    std::vector<std::vector<std::pair<float, uint32_t>>> results(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        args[t] = {t, num_threads, query, base, base_number, vecdim, k, rerank_k,
                   lut, &barrier, &results[t]};
    }

    if (pool) {
        pool->dispatch([&](int t) { pq_pthread_worker(&args[t]); });
    } else {
        std::vector<pthread_t> tids(num_threads);
        for (int t = 0; t < num_threads; ++t)
            pthread_create(&tids[t], nullptr, pq_pthread_worker, &args[t]);
        for (int t = 0; t < num_threads; ++t)
            pthread_join(tids[t], nullptr);
    }
    pthread_barrier_destroy(&barrier);

    // 全局归并
    std::priority_queue<std::pair<float, uint32_t>> final_q;
    for (const auto& tr : results) {
        for (const auto& r : tr) {
            if (final_q.size() < k || r.first < final_q.top().first) {
                final_q.push(r);
                if (final_q.size() > k) final_q.pop();
            }
        }
    }
    return final_q;
}