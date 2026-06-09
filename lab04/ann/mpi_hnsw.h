#pragma once

#include <mpi.h>
#include <vector>
#include <queue>
#include <cstdint>
#include <omp.h>

#include "ivf.h"          // build_ivf, ivf_ns, ivf_neon_ip
#include "hnsw.h"         // build_hnsw / build_hnsw_sub / hnsw_search_baseline / hnsw_search_sub_omp
#include "mpi_common.h"
#include "mpi_ivf.h"      // mpi_ivf_build_local (复用分片+物化)

// C11 / C12 
// 离线: 在本 rank 分片上建 HNSW (C11: 单图; C12: T 个子图)
inline void
mpi_hnsw_build_local(const float* full_base, size_t N, size_t vecdim,
                     PartitionScheme part, int rank, int P,
                     int n_sub /*C11传1, C12传T*/,
                     size_t M, size_t ef_c, int ef_s,
                     std::vector<uint32_t>& shard_ids /*out*/)
{
    omp_set_num_threads(1);
    shard_ids = mpi_compute_shard_ids(N, rank, P, part);
    float* local_base = mpi_materialize_local_base(full_base, vecdim, shard_ids);
    if (n_sub <= 1) build_hnsw    (local_base, shard_ids.size(), vecdim, M, ef_c, ef_s);
    else            build_hnsw_sub(local_base, shard_ids.size(), vecdim, n_sub, M, ef_c, ef_s);
    delete[] local_base;   // hnswlib addPoint 已复制数据
}

// C11: 单查询 单图 searchKnn - 映射全局 id - Gather 归并
inline std::priority_queue<std::pair<float, uint32_t>>
mpi_hnsw_gather(float* q, size_t vecdim, size_t k, size_t local_topk,
                const std::vector<uint32_t>& shard_ids, int rank, int P)
{
    auto local = hnsw_search_baseline(q, vecdim, local_topk);   // (dist, 局部 label)
    return mpi_gather_merge(local, shard_ids, k, local_topk, rank, P);
}

// C12: 单查询 进程内 T 子图 OMP 搜 - 映射全局 id - Gather 归并
inline std::priority_queue<std::pair<float, uint32_t>>
mpi_hnsw_gather_sub(float* q, size_t vecdim, size_t k, size_t local_topk,
                    const std::vector<uint32_t>& shard_ids, int rank, int P, int n_sub)
{
    auto local = hnsw_search_sub_omp(q, vecdim, local_topk, n_sub);  // 已在进程内归并, (dist, 局部 id)
    return mpi_gather_merge(local, shard_ids, k, local_topk, rank, P);
}

// C13 / C14 
namespace mpi_ivfhnsw_ns {
    hnswlib::InnerProductSpace*                   cl_space  = nullptr;   // 簇内 HNSW 共用 space
    std::vector<hnswlib::HierarchicalNSW<float>*> cluster_idx;          // [nlist], 非本 rank 持有的为 nullptr
    hnswlib::InnerProductSpace*                   rt_space  = nullptr;   // 顶层路由 HNSW (C14)
    hnswlib::HierarchicalNSW<float>*              router    = nullptr;
    int nlist = 0;
}

// 离线: 全局 IVF (复制) + 每簇 HNSW (按 c%P 分给各 rank) + 可选顶层路由 HNSW
inline void
mpi_ivfhnsw_build(const float* full_base, size_t N, size_t vecdim, int nlist,
                  int rank, int P, size_t M, size_t ef_c, int ef_s, bool build_router)
{
    using namespace mpi_ivfhnsw_ns;
    omp_set_num_threads(1);

    build_ivf((float*)full_base, N, vecdim, nlist);   // 全局 IVF (各 rank 相同)
    mpi_ivfhnsw_ns::nlist = nlist;

    cl_space = new hnswlib::InnerProductSpace(vecdim);
    cluster_idx.assign(nlist, nullptr);
    for (int c = 0; c < nlist; ++c) {
        if (c % P != rank) continue;                   // 本 rank 只持有 c%P==rank 的簇
        size_t s = ivf_ns::cluster_starts[c], e = ivf_ns::cluster_starts[c + 1];
        size_t cn = e - s;
        if (cn == 0) continue;
        auto* idx = new hnswlib::HierarchicalNSW<float>(cl_space, cn, M, ef_c);
        for (size_t i = s; i < e; ++i)                 // label 用全局原始 id
            idx->addPoint(ivf_ns::reordered_base + i * vecdim,
                          (hnswlib::labeltype)ivf_ns::reordered_ids[i]);
        idx->setEf(ef_s);
        cluster_idx[c] = idx;
    }

    if (build_router) {                                 // C14: 在 nlist 个簇心上建顶层 HNSW
        rt_space = new hnswlib::InnerProductSpace(vecdim);
        router   = new hnswlib::HierarchicalNSW<float>(rt_space, (size_t)nlist, M, ef_c);
        for (int c = 0; c < nlist; ++c)
            router->addPoint(ivf_ns::centroids + (size_t)c * vecdim, (hnswlib::labeltype)c);
        router->setEf(std::max(64, 2 * 16));            // 路由 ef, 略大于常用 nprobe
    }
}

inline void mpi_ivfhnsw_free() {
    using namespace mpi_ivfhnsw_ns;
    for (auto* p : cluster_idx) if (p) delete p;
    cluster_idx.clear();
    if (cl_space) { delete cl_space; cl_space = nullptr; }
    if (router)   { delete router;   router   = nullptr; }
    if (rt_space) { delete rt_space; rt_space = nullptr; }
    free_ivf();
}

// 粗排选 nprobe 个候选簇: use_router=false 暴力扫簇心(C13); true 用顶层 HNSW(C14)
static inline std::vector<int>
mpi_ivfhnsw_coarse(float* q, size_t vecdim, int nprobe, bool use_router)
{
    using namespace mpi_ivfhnsw_ns;
    std::vector<int> probe;
    if (use_router && router) {
        auto raw = router->searchKnn((void*)q, (size_t)nprobe);
        while (!raw.empty()) { probe.push_back((int)raw.top().second); raw.pop(); }
    } else {
        std::priority_queue<std::pair<float, int>> coarse;   // max-heap of (dis, c)
        for (int c = 0; c < nlist; ++c) {
            float dis = 1.0f - ivf_neon_ip(q, ivf_ns::centroids + (size_t)c * vecdim, vecdim);
            if ((int)coarse.size() < nprobe || dis < coarse.top().first) {
                coarse.push({dis, c});
                if ((int)coarse.size() > nprobe) coarse.pop();
            }
        }
        while (!coarse.empty()) { probe.push_back(coarse.top().second); coarse.pop(); }
    }
    return probe;
}

// C13/C14: 单查询 粗排 - 搜本 rank 持有的候选簇 HNSW - Gather 归并 (id 已全局)
inline std::priority_queue<std::pair<float, uint32_t>>
mpi_ivfhnsw_search(float* q, size_t vecdim, size_t k, int nprobe, size_t local_topk,
                   int rank, int P, bool use_router)
{
    using namespace mpi_ivfhnsw_ns;
    std::vector<int> probe = mpi_ivfhnsw_coarse(q, vecdim, nprobe, use_router);

    std::priority_queue<std::pair<float, uint32_t>> local;   // (dist, 全局 id)
    for (int c : probe) {
        auto* idx = cluster_idx[(size_t)c];
        if (!idx) continue;                                  // 非本 rank 持有
        auto raw = idx->searchKnn((void*)q, local_topk);
        while (!raw.empty()) {
            float d = raw.top().first; uint32_t id = (uint32_t)raw.top().second; raw.pop();
            if (local.size() < local_topk || d < local.top().first) {
                local.push({d, id});
                if (local.size() > local_topk) local.pop();
            }
        }
    }
    return mpi_gather_merge_global(local, k, local_topk, rank, P);
}
