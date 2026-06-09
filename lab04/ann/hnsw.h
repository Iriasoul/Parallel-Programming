#pragma once

#include <queue>
#include <vector>
#include <unordered_set>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <omp.h>
#include <pthread.h>
#include "hnswlib/hnswlib/hnswlib.h"

using hnsw_tableint = unsigned int;
using hnsw_distfunc = hnswlib::DISTFUNC<float>;

namespace hnsw_ns {
    hnswlib::InnerProductSpace*      space = nullptr;
    hnswlib::HierarchicalNSW<float>* idx   = nullptr;
    size_t N_elements = 0;
    int    ef_search  = 64;
}

// 离线: 构建单个全局 HNSW 索引
inline void build_hnsw(float* base, size_t N, size_t vecdim,
                       size_t M = 16, size_t ef_c = 200, int ef_s = 64)
{
    using namespace hnsw_ns;
    N_elements = N;
    ef_search  = ef_s;
    space = new hnswlib::InnerProductSpace(vecdim);
    idx   = new hnswlib::HierarchicalNSW<float>(space, N, M, ef_c);
    // hnswlib addPoint 内部有元素级锁, 支持并发调用
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i)
        idx->addPoint(base + i * vecdim, (hnswlib::labeltype)i);
    idx->setEf(ef_s);
}

inline void free_hnsw() {
    using namespace hnsw_ns;
    if (idx)   { delete idx;   idx   = nullptr; }
    if (space) { delete space; space = nullptr; }
    N_elements = 0;
}

// 底层贪心搜索 
static inline std::priority_queue<std::pair<float, uint32_t>>
hnsw_greedy_search_l0(
    const float* query,
    hnsw_tableint entry_point,
    size_t ef, size_t k,
    hnsw_distfunc dist_fn, void* dist_param)
{
    using namespace hnsw_ns;
    using PFT = std::pair<float, hnsw_tableint>;

    // W: 结果集 max-heap, 容量 ef (top = 当前最差候选)
    std::priority_queue<PFT> W;
    // C: 探索集 min-heap (top = 最近的未完全探索节点)
    std::priority_queue<PFT, std::vector<PFT>, std::greater<PFT>> C;

    std::unordered_set<hnsw_tableint> visited;
    visited.reserve(ef * 4);

    auto get_data = [&](hnsw_tableint id) -> const float* {
        return reinterpret_cast<const float*>(
            idx->data_level0_memory_ + (size_t)id * idx->size_data_per_element_
            + idx->offsetData_);
    };
    auto get_links = [&](hnsw_tableint id) -> const char* {
        return idx->data_level0_memory_ + (size_t)id * idx->size_data_per_element_
               + idx->offsetLevel0_;
    };
    auto get_label = [&](hnsw_tableint id) -> uint32_t {
        return (uint32_t)*reinterpret_cast<const hnswlib::labeltype*>(
            idx->data_level0_memory_ + (size_t)id * idx->size_data_per_element_
            + idx->label_offset_);
    };

    float ep_dist = dist_fn((void*)query, (void*)get_data(entry_point), dist_param);
    W.push(std::make_pair(ep_dist, entry_point));
    C.push(std::make_pair(ep_dist, entry_point));
    visited.insert(entry_point);

    while (!C.empty()) {
        float d_closest = C.top().first;
        hnsw_tableint v = C.top().second;
        C.pop();
        if (!W.empty() && d_closest > W.top().first) break;

        const char* ll = get_links(v);
        int num_links = *reinterpret_cast<const int*>(ll);
        const hnsw_tableint* nbs = reinterpret_cast<const hnsw_tableint*>(ll + sizeof(int));

        for (int i = 0; i < num_links; ++i) {
            hnsw_tableint nb = nbs[i];
            if (visited.count(nb)) continue;
            visited.insert(nb);
            float d = dist_fn((void*)query, (void*)get_data(nb), dist_param);
            if ((int)W.size() < (int)ef || d < W.top().first) {
                C.push(std::make_pair(d, nb));
                W.push(std::make_pair(d, nb));
                if ((int)W.size() > (int)ef) W.pop();
            }
        }
    }

    // 从 W 提取 top-k, 转为 (dist, original_label)
    std::priority_queue<std::pair<float, uint32_t>> result;
    while (!W.empty()) {
        float d        = W.top().first;
        uint32_t label = get_label(W.top().second);
        W.pop();
        if (result.size() < k || d < result.top().first) {
            result.push(std::make_pair(d, label));
            if (result.size() > k) result.pop();
        }
    }
    return result;
}

// 全局 merge: 把 T 个局部 top-k 合并为 final top-k
static inline std::priority_queue<std::pair<float, uint32_t>>
hnsw_merge_results(
    std::vector<std::priority_queue<std::pair<float, uint32_t>>>& locals,
    size_t k)
{
    std::priority_queue<std::pair<float, uint32_t>> final_q;
    for (auto& q : locals) {
        while (!q.empty()) {
            auto t = q.top(); q.pop();
            if (final_q.size() < k || t.first < final_q.top().first) {
                final_q.push(t);
                if (final_q.size() > k) final_q.pop();
            }
        }
    }
    return final_q;
}

// S16: HNSW baseline — 直接用 hnswlib searchKnn (单线程)
inline std::priority_queue<std::pair<float, uint32_t>>
hnsw_search_baseline(float* query, size_t /*vecdim*/, size_t k)
{
    using namespace hnsw_ns;
    auto raw = idx->searchKnn((void*)query, k);
    std::priority_queue<std::pair<float, uint32_t>> res;
    while (!raw.empty()) {
        res.push(std::make_pair(raw.top().first, (uint32_t)raw.top().second));
        raw.pop();
    }
    return res;
}

// S 17: HNSW + OpenMP — 切换为子图并行方案 
// S 18: HNSW + Pthread 


// 多入口实验实现
inline std::priority_queue<std::pair<float, uint32_t>>
hnsw_search_multi_entry_omp(float* query, size_t /*vecdim*/, size_t k,
                             int num_threads = 0)
{
    using namespace hnsw_ns;
    int nthr = (num_threads > 0) ? num_threads : omp_get_max_threads();
    hnsw_distfunc dist_fn    = space->get_dist_func();
    void*         dist_param = space->get_dist_func_param();
    size_t N = (size_t)idx->cur_element_count;
    size_t ef = (size_t)ef_search;
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_res(nthr);
    #pragma omp parallel for schedule(static) num_threads(nthr)
    for (int t = 0; t < nthr; ++t) {
        size_t seg = N / (size_t)nthr;
        hnsw_tableint ep = (hnsw_tableint)((size_t)t * seg + seg / 2);
        if (ep >= (hnsw_tableint)N) ep = 0;
        local_res[t] = hnsw_greedy_search_l0(query, ep, ef, k, dist_fn, dist_param);
    }
    return hnsw_merge_results(local_res, k);
}

struct HNSWPthreadArgs {
    const float*     query;
    hnsw_tableint    entry_point;
    size_t           ef;
    size_t           k;
    hnsw_distfunc    dist_fn;
    void*            dist_param;
    std::priority_queue<std::pair<float, uint32_t>>* result;
};

inline void* hnsw_pthread_worker(void* arg) {
    HNSWPthreadArgs* a = (HNSWPthreadArgs*)arg;
    *a->result = hnsw_greedy_search_l0(
        a->query, a->entry_point, a->ef, a->k, a->dist_fn, a->dist_param);
    return nullptr;
}

// 多子图方案 (仅用 hnswlib 公开 API, 不需要访问内部字段)
namespace hnsw_sub_ns {
    std::vector<hnswlib::InnerProductSpace*>      sub_spaces;
    std::vector<hnswlib::HierarchicalNSW<float>*> sub_idx;
    std::vector<std::vector<uint32_t>>            sub_orig_ids;
    int ef_search = 64;
}

inline void build_hnsw_sub(float* base, size_t N, size_t vecdim,
                            int num_sub = 4, size_t M = 16,
                            size_t ef_c = 200, int ef_s = 64)
{
    using namespace hnsw_sub_ns;
    ef_search = ef_s;
    sub_spaces.resize(num_sub, nullptr);
    sub_idx.resize(num_sub, nullptr);
    sub_orig_ids.resize(num_sub);

    std::vector<int> assign(N);
    for (size_t i = 0; i < N; ++i) assign[i] = (int)(i % num_sub);
    // simple Fisher-Yates shuffle
    for (size_t i = N - 1; i > 0; --i) {
        size_t j = ((size_t)rand() * rand()) % (i + 1);
        std::swap(assign[i], assign[j]);
    }
    for (size_t i = 0; i < N; ++i)
        sub_orig_ids[(size_t)assign[i]].push_back((uint32_t)i);

    #pragma omp parallel for schedule(dynamic, 1) num_threads(num_sub)
    for (int s = 0; s < num_sub; ++s) {
        size_t sub_n = sub_orig_ids[(size_t)s].size();
        sub_spaces[(size_t)s] = new hnswlib::InnerProductSpace(vecdim);
        sub_idx[(size_t)s]    = new hnswlib::HierarchicalNSW<float>(
                                    sub_spaces[(size_t)s], sub_n, M, ef_c);
        for (size_t j = 0; j < sub_n; ++j)
            sub_idx[(size_t)s]->addPoint(
                base + (size_t)sub_orig_ids[(size_t)s][j] * vecdim, j);
        sub_idx[(size_t)s]->setEf(ef_s);
    }
}

inline void free_hnsw_sub() {
    using namespace hnsw_sub_ns;
    for (auto* p : sub_idx)    if (p) delete p;
    for (auto* p : sub_spaces) if (p) delete p;
    sub_idx.clear(); sub_spaces.clear(); sub_orig_ids.clear();
}

struct HNSWSubArgs {
    float* query;
    size_t k;
    int    sub_id;
    std::priority_queue<std::pair<float, uint32_t>>* result;
};

inline void* hnsw_sub_pthread_worker(void* arg) {
    HNSWSubArgs* a = (HNSWSubArgs*)arg;
    using namespace hnsw_sub_ns;
    int s = a->sub_id;
    auto raw = sub_idx[(size_t)s]->searchKnn((void*)a->query, a->k);
    while (!raw.empty()) {
        float    d    = raw.top().first;
        uint32_t orig = sub_orig_ids[(size_t)s][(size_t)raw.top().second];
        raw.pop();
        if (a->result->size() < a->k || d < a->result->top().first) {
            a->result->push(std::make_pair(d, orig));
            if (a->result->size() > a->k) a->result->pop();
        }
    }
    return nullptr;
}

inline std::priority_queue<std::pair<float, uint32_t>>
hnsw_search_sub_omp(float* query, size_t /*vecdim*/, size_t k,
                    int num_threads = 0)
{
    using namespace hnsw_sub_ns;
    int nthr = (num_threads > 0) ? num_threads : (int)sub_idx.size();
    nthr = std::min(nthr, (int)sub_idx.size());
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_res((size_t)nthr);

    #pragma omp parallel for schedule(static) num_threads(nthr)
    for (int s = 0; s < nthr; ++s) {
        auto raw = sub_idx[(size_t)s]->searchKnn((void*)query, k);
        while (!raw.empty()) {
            float    d    = raw.top().first;
            uint32_t orig = sub_orig_ids[(size_t)s][(size_t)raw.top().second];
            raw.pop();
            if (local_res[(size_t)s].size() < k || d < local_res[(size_t)s].top().first) {
                local_res[(size_t)s].push(std::make_pair(d, orig));
                if (local_res[(size_t)s].size() > k) local_res[(size_t)s].pop();
            }
        }
    }
    std::priority_queue<std::pair<float, uint32_t>> final_q;
    for (auto& q : local_res) {
        while (!q.empty()) {
            auto t = q.top(); q.pop();
            if (final_q.size() < k || t.first < final_q.top().first) {
                final_q.push(t);
                if (final_q.size() > k) final_q.pop();
            }
        }
    }
    return final_q;
}

inline std::priority_queue<std::pair<float, uint32_t>>
hnsw_search_sub_pthread(float* query, size_t /*vecdim*/, size_t k,
                        int num_threads = 4)
{
    using namespace hnsw_sub_ns;
    int nthr = std::min(num_threads, (int)sub_idx.size());
    std::vector<pthread_t>  tids((size_t)nthr);
    std::vector<HNSWSubArgs> args((size_t)nthr);
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_res((size_t)nthr);

    for (int s = 0; s < nthr; ++s) {
        args[(size_t)s].query  = query;
        args[(size_t)s].k      = k;
        args[(size_t)s].sub_id = s;
        args[(size_t)s].result = &local_res[(size_t)s];
        pthread_create(&tids[(size_t)s], nullptr,
                       hnsw_sub_pthread_worker, &args[(size_t)s]);
    }
    for (int s = 0; s < nthr; ++s) pthread_join(tids[(size_t)s], nullptr);

    std::priority_queue<std::pair<float, uint32_t>> final_q;
    for (auto& q : local_res) {
        while (!q.empty()) {
            auto t = q.top(); q.pop();
            if (final_q.size() < k || t.first < final_q.top().first) {
                final_q.push(t);
                if (final_q.size() > k) final_q.pop();
            }
        }
    }
    return final_q;
}

// wrapper 
inline std::priority_queue<std::pair<float, uint32_t>>
hnsw_search_omp(float* query, size_t vecdim, size_t k,
                int num_threads = 0)
{
    return hnsw_search_sub_omp(query, vecdim, k, num_threads);
}

