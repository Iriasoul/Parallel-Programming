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

#include "ivf.h"  // 复用 IVF 索引 (centroids, reordered_base, ids, cluster_starts)

namespace ivfpq2_ns {
    int PQM = 8;
    const int PQK = 256;
    // 每簇独立的 PQ 模型. 用 vector<指针> 维护变长结构.
    // per_cluster_centroids[c]: [PQM][PQK][sub_dim] (KMeans 训练得到的原始布局)
    // per_cluster_centroids_soa[c]: [PQM][sub_dim][PQK] (LUT 构建的 SIMD 友好布局)
    // per_cluster_codes[c]: 该簇所有向量的 PQ 编码, 长度 = cluster_size * PQM (AoS)
    // per_cluster_codes_soa[c]: SoA, 长度 = PQM * cluster_size, [m * size + i]
    std::vector<float*>   per_cluster_centroids;
    std::vector<float*>   per_cluster_centroids_soa;
    std::vector<uint8_t*> per_cluster_codes;
    std::vector<uint8_t*> per_cluster_codes_soa;
}

// 内积 
static inline float ivfpq2_neon_ip(const float* a, const float* b, size_t dim) {
    float32x4_t s = vdupq_n_f32(0.0f);
    size_t d = 0;
    for (; d + 3 < dim; d += 4) {
        s = vfmaq_f32(s, vld1q_f32(a + d), vld1q_f32(b + d));
    }
    float sum = vaddvq_f32(s);
    for (; d < dim; ++d) sum += a[d] * b[d];
    return sum;
}

// 簇内 KMeans, 内积聚类
inline void ivfpq2_kmeans_subspace(const float* data, size_t n, size_t dim,
                                   float* cents, int K, int iters = 20)
{
    if (n == 0) {
        std::memset(cents, 0, K * dim * sizeof(float));
        return;
    }
    // 初始化
    for (int i = 0; i < K; ++i) {
        size_t src = (n >= (size_t)K) ? (rand() % n) : (i % n);
        std::memcpy(cents + i * dim, data + src * dim, dim * sizeof(float));
    }
    if (n < (size_t)K) return;  // 数据量不够, 不再迭代 (centroids 已退化为部分原始向量)

    for (int it = 0; it < iters; ++it) {
        std::vector<float> next(K * dim, 0);
        std::vector<int>   cnt(K, 0);
        for (size_t i = 0; i < n; ++i) {
            float best_ip = -1e30f; int best = 0;
            for (int c = 0; c < K; ++c) {
                float ip = ivfpq2_neon_ip(data + i * dim, cents + c * dim, dim);
                if (ip > best_ip) { best_ip = ip; best = c; }
            }
            cnt[best]++;
            for (size_t d = 0; d < dim; ++d) next[best * dim + d] += data[i * dim + d];
        }
        for (int c = 0; c < K; ++c) {
            if (cnt[c]) for (size_t d = 0; d < dim; ++d) cents[c * dim + d] = next[c * dim + d] / cnt[c];
        }
    }
}

// 离线: 在 IVF 重排后的 base 上, 对每个簇独立训练 PQ
inline void build_ivf_pq_v2(size_t base_number, size_t vecdim, int pq_m) {
    using namespace ivfpq2_ns;
    using namespace ivf_ns;
    (void)base_number;  // 与方法 1 接口对齐, 但本实现通过 cluster_starts 定簇范围, 不需要全库数
    PQM = pq_m;
    size_t sub_dim = vecdim / PQM;

    per_cluster_centroids.assign(nlist, nullptr);
    per_cluster_centroids_soa.assign(nlist, nullptr);
    per_cluster_codes.assign(nlist, nullptr);
    per_cluster_codes_soa.assign(nlist, nullptr);

    // 簇间外层并行 (不同簇互相独立)
    #pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < nlist; ++c) {
        size_t s = cluster_starts[c];
        size_t e = cluster_starts[c + 1];
        size_t csize = e - s;

        per_cluster_centroids[c]     = new float[PQM * PQK * sub_dim];
        per_cluster_centroids_soa[c] = new float[PQM * sub_dim * PQK];
        if (csize > 0) {
            per_cluster_codes[c]     = new uint8_t[csize * PQM];
            per_cluster_codes_soa[c] = new uint8_t[PQM * csize];
        }

        // 对每个子空间, 用簇内向量训练 PQK 个 centroids
        for (int m = 0; m < PQM; ++m) {
            std::vector<float> sub(csize * sub_dim);
            for (size_t i = 0; i < csize; ++i) {
                std::memcpy(&sub[i * sub_dim],
                            reordered_base + (s + i) * vecdim + m * sub_dim,
                            sub_dim * sizeof(float));
            }
            ivfpq2_kmeans_subspace(sub.data(), csize, sub_dim,
                                   per_cluster_centroids[c] + m * PQK * sub_dim,
                                   PQK, 15);
        }

        // 簇内每个向量用本簇 PQ 模型编码
        for (size_t i = 0; i < csize; ++i) {
            for (int m = 0; m < PQM; ++m) {
                float* sub_v = reordered_base + (s + i) * vecdim + m * sub_dim;
                float* cents = per_cluster_centroids[c] + m * PQK * sub_dim;
                float best_ip = -1e30f; int best = 0;
                for (int kk = 0; kk < PQK; ++kk) {
                    float ip = ivfpq2_neon_ip(sub_v, cents + kk * sub_dim, sub_dim);
                    if (ip > best_ip) { best_ip = ip; best = kk; }
                }
                per_cluster_codes[c][i * PQM + m] = (uint8_t)best;
            }
        }

        // SoA 重排
        for (int m = 0; m < PQM; ++m) {
            for (size_t i = 0; i < csize; ++i) {
                per_cluster_codes_soa[c][m * csize + i] = per_cluster_codes[c][i * PQM + m];
            }
        }
        // centroids 也建 SoA: [PQM][sub_dim][PQK], LUT 构建用
        for (int m = 0; m < PQM; ++m) {
            for (int d = 0; d < (int)sub_dim; ++d) {
                for (int kk = 0; kk < PQK; ++kk) {
                    per_cluster_centroids_soa[c][m * sub_dim * PQK + d * PQK + kk] =
                        per_cluster_centroids[c][m * PQK * sub_dim + kk * sub_dim + d];
                }
            }
        }
    }
}

inline void free_ivf_pq_v2() {
    using namespace ivfpq2_ns;
    for (auto* p : per_cluster_centroids)     if (p) delete[] p;
    for (auto* p : per_cluster_centroids_soa) if (p) delete[] p;
    for (auto* p : per_cluster_codes)         if (p) delete[] p;
    for (auto* p : per_cluster_codes_soa)     if (p) delete[] p;
    per_cluster_centroids.clear();
    per_cluster_centroids_soa.clear();
    per_cluster_codes.clear();
    per_cluster_codes_soa.clear();
}

// 构建该簇 LUT
static inline void ivfpq2_build_lut_for_cluster(const float* query, size_t sub_dim,
                                                int c, float lut[][256])
{
    using namespace ivfpq2_ns;
    for (int m = 0; m < PQM; ++m) {
        const float* sub_q = query + m * sub_dim;
        const float* m_soa = per_cluster_centroids_soa[c] + m * sub_dim * PQK;
        for (int kk = 0; kk < PQK; kk += 4) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (size_t d = 0; d < sub_dim; ++d) {
                acc = vfmaq_f32(acc, vdupq_n_f32(sub_q[d]),
                                vld1q_f32(m_soa + d * PQK + kk));
            }
            vst1q_f32(&lut[m][kk], acc);
        }
    }
}

// S13: IVF-PQ-M2 单线程
inline std::priority_queue<std::pair<float, uint32_t>>
ivf_pq2_search_simd(float* query, size_t vecdim,
                    size_t k, int nprobe, size_t rerank_k)
{
    using namespace ivf_ns;
    using namespace ivfpq2_ns;
    size_t sub_dim = vecdim / PQM;

    // IVF 粗排: 选 nprobe 个簇
    std::priority_queue<std::pair<float, int>> coarse;
    for (int c = 0; c < nlist; ++c) {
        float ip = ivfpq2_neon_ip(query, centroids + c * vecdim, vecdim);
        float dis = 1.0f - ip;
        if ((int)coarse.size() < nprobe || dis < coarse.top().first) {
            coarse.push({dis, c});
            if ((int)coarse.size() > nprobe) coarse.pop();
        }
    }
    std::vector<int> probe_list;
    probe_list.reserve(nprobe);
    while (!coarse.empty()) { probe_list.push_back(coarse.top().second); coarse.pop(); }

    // 每访问一簇都重建 LUT 然后 ADC 粗排
    alignas(64) float lut[64][256];
    std::priority_queue<std::pair<float, uint32_t>> coarse_q;
    for (int c : probe_list) {
        ivfpq2_build_lut_for_cluster(query, sub_dim, c, lut);

        size_t s = cluster_starts[c];
        size_t e = cluster_starts[c + 1];
        size_t csize = e - s;
        const uint8_t* codes_soa = per_cluster_codes_soa[c];
        for (size_t i = 0; i < csize; ++i) {
            float ip = 0.0f;
            for (int m = 0; m < PQM; ++m) {
                ip += lut[m][codes_soa[m * csize + i]];
            }
            float dis = 1.0f - ip;
            // 这里存 reordered 位置 (s + i), 与方法 1 保持一致便于精排
            if (coarse_q.size() < rerank_k || dis < coarse_q.top().first) {
                coarse_q.push({dis, (uint32_t)(s + i)});
                if (coarse_q.size() > rerank_k) coarse_q.pop();
            }
        }
    }

    // 精排
    std::priority_queue<std::pair<float, uint32_t>> final_q;
    while (!coarse_q.empty()) {
        uint32_t pos = coarse_q.top().second;
        coarse_q.pop();
        float ip = ivfpq2_neon_ip(query, reordered_base + pos * vecdim, vecdim);
        float dis = 1.0f - ip;
        if (final_q.size() < k || dis < final_q.top().first) {
            final_q.push({dis, reordered_ids[pos]});
            if (final_q.size() > k) final_q.pop();
        }
    }
    return final_q;
}

// S14: IVF-PQ-M2 + OpenMP
inline std::priority_queue<std::pair<float, uint32_t>>
ivf_pq2_search_omp(float* query, size_t vecdim,
                   size_t k, int nprobe, size_t rerank_k,
                   int num_threads = 0)
{
    using namespace ivf_ns;
    using namespace ivfpq2_ns;
    size_t sub_dim = vecdim / PQM;

    int nthr = (num_threads > 0) ? num_threads : omp_get_max_threads();

    // 粗排单线程
    std::priority_queue<std::pair<float, int>> coarse;
    for (int c = 0; c < nlist; ++c) {
        float ip = ivfpq2_neon_ip(query, centroids + c * vecdim, vecdim);
        float dis = 1.0f - ip;
        if ((int)coarse.size() < nprobe || dis < coarse.top().first) {
            coarse.push({dis, c});
            if ((int)coarse.size() > nprobe) coarse.pop();
        }
    }
    std::vector<int> probe_list;
    probe_list.reserve(nprobe);
    while (!coarse.empty()) { probe_list.push_back(coarse.top().second); coarse.pop(); }

    // 簇级 dynamic, 每线程独立局部粗排 + 局部精排
    std::vector<std::vector<std::pair<float, uint32_t>>> thread_results(nthr);

    #pragma omp parallel num_threads(nthr)
    {
        int tid = omp_get_thread_num();
        alignas(64) float lut[64][256];  // 线程私有 LUT
        std::priority_queue<std::pair<float, uint32_t>> local_coarse;

        #pragma omp for schedule(dynamic, 1) nowait
        for (int pi = 0; pi < (int)probe_list.size(); ++pi) {
            int c = probe_list[pi];
            ivfpq2_build_lut_for_cluster(query, sub_dim, c, lut);

            size_t s = cluster_starts[c];
            size_t e = cluster_starts[c + 1];
            size_t csize = e - s;
            const uint8_t* codes_soa = per_cluster_codes_soa[c];
            for (size_t i = 0; i < csize; ++i) {
                float ip = 0.0f;
                for (int m = 0; m < PQM; ++m) {
                    ip += lut[m][codes_soa[m * csize + i]];
                }
                float dis = 1.0f - ip;
                if (local_coarse.size() < rerank_k || dis < local_coarse.top().first) {
                    local_coarse.push({dis, (uint32_t)(s + i)});
                    if (local_coarse.size() > rerank_k) local_coarse.pop();
                }
            }
        }

        // 线程内精排
        std::priority_queue<std::pair<float, uint32_t>> local_fine;
        while (!local_coarse.empty()) {
            uint32_t pos = local_coarse.top().second;
            local_coarse.pop();
            float ip = ivfpq2_neon_ip(query, reordered_base + pos * vecdim, vecdim);
            float dis = 1.0f - ip;
            uint32_t orig_id = reordered_ids[pos];
            if (local_fine.size() < k || dis < local_fine.top().first) {
                local_fine.push({dis, orig_id});
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

// S15: IVF-PQ-v2 + Pthread
struct IVFPQ2PthreadArgs {
    int tid;
    float* query;
    size_t vecdim;
    size_t k;
    size_t rerank_k;
    int* probe_list;
    int nprobe;
    int* next_task_idx;
    pthread_mutex_t* mtx;
    std::vector<std::pair<float, uint32_t>>* result;
};

inline void* ivf_pq2_pthread_worker(void* arg) {
    IVFPQ2PthreadArgs* a = (IVFPQ2PthreadArgs*)arg;
    using namespace ivf_ns;
    using namespace ivfpq2_ns;
    size_t sub_dim = a->vecdim / PQM;

    alignas(64) float lut[64][256];  // 线程私有
    std::priority_queue<std::pair<float, uint32_t>> local_coarse;

    while (true) {
        pthread_mutex_lock(a->mtx);
        int task = (*a->next_task_idx)++;
        pthread_mutex_unlock(a->mtx);
        if (task >= a->nprobe) break;

        int c = a->probe_list[task];
        ivfpq2_build_lut_for_cluster(a->query, sub_dim, c, lut);

        size_t s = cluster_starts[c];
        size_t e = cluster_starts[c + 1];
        size_t csize = e - s;
        const uint8_t* codes_soa = per_cluster_codes_soa[c];
        for (size_t i = 0; i < csize; ++i) {
            float ip = 0.0f;
            for (int m = 0; m < PQM; ++m) {
                ip += lut[m][codes_soa[m * csize + i]];
            }
            float dis = 1.0f - ip;
            if (local_coarse.size() < a->rerank_k || dis < local_coarse.top().first) {
                local_coarse.push({dis, (uint32_t)(s + i)});
                if (local_coarse.size() > a->rerank_k) local_coarse.pop();
            }
        }
    }

    // 线程内精排
    std::priority_queue<std::pair<float, uint32_t>> local_fine;
    while (!local_coarse.empty()) {
        uint32_t pos = local_coarse.top().second;
        local_coarse.pop();
        float ip = ivfpq2_neon_ip(a->query, reordered_base + pos * a->vecdim, a->vecdim);
        float dis = 1.0f - ip;
        uint32_t orig_id = reordered_ids[pos];
        if (local_fine.size() < a->k || dis < local_fine.top().first) {
            local_fine.push({dis, orig_id});
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
