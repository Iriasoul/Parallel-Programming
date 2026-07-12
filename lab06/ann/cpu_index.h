// cpu_index.h 
// CPU : IVF / IVF-PQ 索引
// 与 lab04 一致；线程内局部粗排+精排后归并
#pragma once
#include "bench_common.h"
#include <omp.h>

// IVF
namespace ivf_ns {
    int nlist = 256;
    float* centroids = nullptr;             // [nlist][dim]
    float* reordered_base = nullptr;        // 同簇连续
    std::vector<uint32_t> reordered_ids;    // 重排位置 -> 原始 id
    std::vector<size_t> cluster_starts;     // 长度 nlist+1
}

inline void ivf_kmeans(const float* data, size_t n, size_t dim, int nlist,
                       float* cents, int iters = 15) {
    for (int i = 0; i < nlist; ++i)
        std::memcpy(cents + i * dim, data + (rand() % n) * dim, dim * sizeof(float));
    std::vector<int> assign(n, 0);
    for (int it = 0; it < iters; ++it) {
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < (long)n; ++i) {
            float best = 1e30f; int bc = 0;
            for (int c = 0; c < nlist; ++c) {
                float d2 = l2sq_default(data + i * dim, cents + c * dim, dim);
                if (d2 < best) { best = d2; bc = c; }
            }
            assign[i] = bc;
        }
        std::vector<float> next(nlist * dim, 0.0f);
        std::vector<int> cnt(nlist, 0);
        for (size_t i = 0; i < n; ++i) {
            int c = assign[i]; cnt[c]++;
            for (size_t d = 0; d < dim; ++d) next[c * dim + d] += data[i * dim + d];
        }
        for (int c = 0; c < nlist; ++c)
            if (cnt[c]) for (size_t d = 0; d < dim; ++d) cents[c * dim + d] = next[c * dim + d] / cnt[c];
    }
}

inline void build_ivf(const float* base, size_t N, size_t dim, int nlist) {
    using namespace ivf_ns;
    ivf_ns::nlist = nlist;
    centroids = new float[nlist * dim];
    size_t train_n = std::min<size_t>(20000, N);
    std::vector<float> train(train_n * dim);
    std::memcpy(train.data(), base, train_n * dim * sizeof(float));
    ivf_kmeans(train.data(), train_n, dim, nlist, centroids, 15);

    std::vector<int> assign(N);
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)N; ++i) {
        float best = 1e30f; int bc = 0;
        for (int c = 0; c < nlist; ++c) {
            float d2 = l2sq_default(base + i * dim, centroids + c * dim, dim);
            if (d2 < best) { best = d2; bc = c; }
        }
        assign[i] = bc;
    }
    cluster_starts.assign(nlist + 1, 0);
    for (size_t i = 0; i < N; ++i) cluster_starts[assign[i] + 1]++;
    for (int c = 1; c <= nlist; ++c) cluster_starts[c] += cluster_starts[c - 1];

    reordered_base = new float[N * dim];
    reordered_ids.assign(N, 0);
    std::vector<size_t> ptr(cluster_starts.begin(), cluster_starts.begin() + nlist);
    for (size_t i = 0; i < N; ++i) {
        int c = assign[i]; size_t pos = ptr[c]++;
        std::memcpy(reordered_base + pos * dim, base + i * dim, dim * sizeof(float));
        reordered_ids[pos] = (uint32_t)i;
    }
}

inline void free_ivf() {
    using namespace ivf_ns;
    delete[] centroids;  centroids = nullptr;
    delete[] reordered_base; reordered_base = nullptr;
    reordered_ids.clear(); cluster_starts.clear();
}

// 粗排：选 nprobe 个最近簇
inline void ivf_select_probes(const float* q, size_t dim, int nprobe, std::vector<int>& probe) {
    using namespace ivf_ns;
    std::priority_queue<std::pair<float,int>> coarse;
    for (int c = 0; c < nlist; ++c) {
        float dis = 1.0f - ip_default(q, centroids + c * dim, dim);
        if ((int)coarse.size() < nprobe || dis < coarse.top().first) {
            coarse.push({dis, c});
            if ((int)coarse.size() > nprobe) coarse.pop();
        }
    }
    probe.clear(); probe.reserve(nprobe);
    while (!coarse.empty()) { probe.push_back(coarse.top().second); coarse.pop(); }
}

// IVF-SIMD 单查询串行版
// 供上层按 query 并行调用，避免嵌套 OMP
inline std::vector<uint32_t>
ivf_search_serial(const float* q, size_t dim, size_t k, int nprobe) {
    using namespace ivf_ns;
    std::vector<int> probe; ivf_select_probes(q, dim, nprobe, probe);
    MaxHeap h;
    for (int c : probe)
        for (size_t i = cluster_starts[c]; i < cluster_starts[c+1]; ++i)
            heap_push(h, 1.0f - ip_default(q, reordered_base + i * dim, dim), reordered_ids[i], k);
    return heap_to_ids(h);
}

// IVF-SIMD + OpenMP（精确簇内扫描）
inline std::vector<uint32_t>
ivf_search_omp(const float* q, size_t dim, size_t k, int nprobe, int nthr) {
    using namespace ivf_ns;
    std::vector<int> probe; ivf_select_probes(q, dim, nprobe, probe);
    std::vector<MaxHeap> local(nthr);
    #pragma omp parallel num_threads(nthr)
    {
        int tid = omp_get_thread_num();
        MaxHeap& h = local[tid];
        #pragma omp for schedule(dynamic,1) nowait
        for (int pi = 0; pi < (int)probe.size(); ++pi) {
            int c = probe[pi];
            for (size_t i = cluster_starts[c]; i < cluster_starts[c+1]; ++i) {
                float dis = 1.0f - ip_default(q, reordered_base + i * dim, dim);
                heap_push(h, dis, reordered_ids[i], k);
            }
        }
    }
    MaxHeap fin;
    for (auto& h : local) while (!h.empty()) { auto t = h.top(); h.pop(); heap_push(fin, t.first, t.second, k); }
    return heap_to_ids(fin);
}

// IVF-PQ
namespace ivfpq_ns {
    int PQM = 8;
    const int PQK = 256;
    uint8_t* pq_codes_soa = nullptr;    // [PQM][N]
    float* pq_centroids = nullptr;      // [PQM][PQK][subdim]
    float* pq_centroids_soa = nullptr;  // [PQM][subdim][PQK]
    size_t g_N = 0;
}

inline void ivfpq_kmeans(const float* data, size_t n, size_t dim, float* cents, int iters = 20) {
    using namespace ivfpq_ns;
    for (int i = 0; i < PQK; ++i)
        std::memcpy(cents + i * dim, data + (rand() % n) * dim, dim * sizeof(float));
    for (int it = 0; it < iters; ++it) {
        std::vector<float> next(PQK * dim, 0); std::vector<int> cnt(PQK, 0);
        for (size_t i = 0; i < n; ++i) {
            float best = 1e30f; int bc = 0;
            for (int c = 0; c < PQK; ++c) {
                float d2 = l2sq_default(data + i * dim, cents + c * dim, dim);
                if (d2 < best) { best = d2; bc = c; }
            }
            cnt[bc]++;
            for (size_t d = 0; d < dim; ++d) next[bc * dim + d] += data[i * dim + d];
        }
        for (int c = 0; c < PQK; ++c)
            if (cnt[c]) for (size_t d = 0; d < dim; ++d) cents[c * dim + d] = next[c * dim + d] / cnt[c];
    }
}

inline void build_ivf_pq(size_t N, size_t dim, int pq_m) {
    using namespace ivfpq_ns; using namespace ivf_ns;
    PQM = pq_m; g_N = N;
    size_t sub = dim / PQM;
    pq_centroids = new float[PQM * PQK * sub];
    size_t train_n = std::min<size_t>(20000, N);
    for (int m = 0; m < PQM; ++m) {
        std::vector<float> subd(train_n * sub);
        for (size_t i = 0; i < train_n; ++i)
            std::memcpy(&subd[i*sub], reordered_base + i*dim + m*sub, sub*sizeof(float));
        ivfpq_kmeans(subd.data(), train_n, sub, pq_centroids + m*PQK*sub, 20);
    }
    std::vector<uint8_t> codes(N * PQM);
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)N; ++i)
        for (int m = 0; m < PQM; ++m) {
            const float* sv = reordered_base + i*dim + m*sub;
            const float* ce = pq_centroids + m*PQK*sub;
            float best = 1e30f; int bc = 0;
            for (int c = 0; c < PQK; ++c) { float d2 = l2sq_default(sv, ce + c*sub, sub); if (d2 < best) { best = d2; bc = c; } }
            codes[i*PQM + m] = (uint8_t)bc;
        }
    pq_codes_soa = new uint8_t[PQM * N];
    for (int m = 0; m < PQM; ++m) for (size_t i = 0; i < N; ++i) pq_codes_soa[m*N + i] = codes[i*PQM + m];
    pq_centroids_soa = new float[PQM * sub * PQK];
    for (int m = 0; m < PQM; ++m) for (int d = 0; d < (int)sub; ++d) for (int c = 0; c < PQK; ++c)
        pq_centroids_soa[m*sub*PQK + d*PQK + c] = pq_centroids[m*PQK*sub + c*sub + d];
}

inline void free_ivf_pq() {
    using namespace ivfpq_ns;
    delete[] pq_codes_soa; pq_codes_soa = nullptr;
    delete[] pq_centroids; pq_centroids = nullptr;
    delete[] pq_centroids_soa; pq_centroids_soa = nullptr;
}

// 建 LUT[m][c] = <q_sub_m, centroid_{m,c}>，跨 centroid 用 AVX2 一次 8 路
static inline void ivfpq_build_lut(const float* q, size_t dim, float lut[][256]) {
    using namespace ivfpq_ns;
    size_t sub = dim / PQM;
    for (int m = 0; m < PQM; ++m) {
        const float* sq = q + m * sub;
        const float* msoa = pq_centroids_soa + m * sub * PQK;
#if BENCH_HAVE_AVX2
        for (int c = 0; c < PQK; c += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (size_t d = 0; d < sub; ++d)
                acc = _mm256_fmadd_ps(_mm256_set1_ps(sq[d]), _mm256_loadu_ps(msoa + d*PQK + c), acc);
            _mm256_storeu_ps(&lut[m][c], acc);
        }
#else
        for (int c = 0; c < PQK; ++c) {
            float s = 0; for (size_t d = 0; d < sub; ++d) s += sq[d] * msoa[d*PQK + c];
            lut[m][c] = s;
        }
#endif
    }
}

// IVF-PQ + OpenMP：LUT -> 簇级 PQ-ADC 局部粗排 -> 线程内 SIMD 精排 -> 归并
inline std::vector<uint32_t>
ivf_pq_search_omp(const float* q, size_t dim, size_t k, int nprobe, size_t rerank_k, int nthr) {
    using namespace ivf_ns; using namespace ivfpq_ns;
    alignas(64) float lut[64][256];
    ivfpq_build_lut(q, dim, lut);
    std::vector<int> probe; ivf_select_probes(q, dim, nprobe, probe);
    size_t N = g_N;

    std::vector<std::vector<std::pair<float,uint32_t>>> tres(nthr);
    #pragma omp parallel num_threads(nthr)
    {
        MaxHeap coarse;   // 局部 PQ-ADC 粗排（保留 rerank_k）
        #pragma omp for schedule(dynamic,1) nowait
        for (int pi = 0; pi < (int)probe.size(); ++pi) {
            int c = probe[pi];
            for (size_t i = cluster_starts[c]; i < cluster_starts[c+1]; ++i) {
                float ip = 0.0f;
                for (int m = 0; m < PQM; ++m) ip += lut[m][pq_codes_soa[m*N + i]];
                heap_push(coarse, 1.0f - ip, (uint32_t)i, rerank_k);
            }
        }
        MaxHeap fine;     // 局部精排
        while (!coarse.empty()) {
            uint32_t pos = coarse.top().second; coarse.pop();
            float dis = 1.0f - ip_default(q, reordered_base + pos * dim, dim);
            heap_push(fine, dis, reordered_ids[pos], k);
        }
        int tid = omp_get_thread_num();
        tres[tid] = heap_to_ids_pairs(fine);
    }
    MaxHeap fin;
    for (auto& tr : tres) for (auto& r : tr) heap_push(fin, r.first, r.second, k);
    return heap_to_ids(fin);
}

// 异构协同用 CPU 工作函数
// 原 cpu_worker.cpp；OpenMP 在 nvcc host 编译下经 -Xcompiler /openmp 正常并行

// 对一批 query 的候选 id 做精确 FP32(AVX2) 内积精排，取 top-k，写入 out[off+i]
// 供融合策略的 L2(CPU 精排) 使用：GPU 回传每条 query 的 rerank_k 候选，再由 CPU 重算精确距离
inline void cpu_rerank_batch(const float* base, const float* query, int d, size_t k,
                             int B, int rerank_k, const uint32_t* cand,
                             std::vector<std::vector<uint32_t>>& out, size_t off, int nthr) {
    #pragma omp parallel for schedule(static) num_threads(nthr)
    for (int i = 0; i < B; ++i) {
        const float* q = query + (size_t)(off + i) * d;
        const uint32_t* c = cand + (size_t)i * rerank_k;
        MaxHeap h;
        for (int j = 0; j < rerank_k; ++j) {
            uint32_t id = c[j];
            heap_push(h, 1.0f - ip_default(q, base + (size_t)id * d, d), id, k);
        }
        out[off + i] = heap_to_ids(h);
    }
}

// 用 IVF 处理 query 区间 [lo,hi)（OpenMP 跨 query 并行，每条走串行 ivf_search_serial）
// 供融合策略的 CPU 分支使用：与 GPU 分支并行处理不相交的 query 区段
inline void cpu_ivf_search_range(const float* query, int d, size_t k, int nprobe,
                                 size_t lo, size_t hi,
                                 std::vector<std::vector<uint32_t>>& out, int nthr) {
    #pragma omp parallel for schedule(dynamic, 8) num_threads(nthr)
    for (long i = (long)lo; i < (long)hi; ++i)
        out[i] = ivf_search_serial(query + (size_t)i * d, d, k, nprobe);
}
