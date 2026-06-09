#pragma once

#include <mpi.h>
#include <vector>
#include <queue>
#include <cstdint>
#include <omp.h>

#include "ivf.h"
#include "ivf_pq.h"
#include "ivf_pq_v2.h"
#include "mpi_common.h"
#include "mpi_ivf.h"     // 复用 mpi_ivf_build_local

// 离线: 分片建 IVF, 再在分片上建 PQ (method=1 全局PQ / method=2 簇内独立PQ)
inline void
mpi_ivfpq_build_local(const float* full_base, size_t N, size_t vecdim,
                      int nlist, int pq_m, int method,
                      PartitionScheme part, int rank, int P,
                      std::vector<uint32_t>& shard_ids /*out*/)
{
    mpi_ivf_build_local(full_base, N, vecdim, nlist, part, rank, P, shard_ids);
    size_t shard_n = shard_ids.size();
    if (method == 1) build_ivf_pq   (shard_n, vecdim, pq_m);   // 依赖 ivf_ns::reordered_base
    else             build_ivf_pq_v2(shard_n, vecdim, pq_m);
}

// 在线单查询: PQ 本地搜索 + Gather 归并 (method 1/2; omp_threads>=2 使用 OpenMP)
inline std::priority_queue<std::pair<float, uint32_t>>
mpi_ivfpq_gather(float* q, size_t vecdim, size_t k, int nprobe, size_t rerank_k,
                 size_t local_topk, const std::vector<uint32_t>& shard_ids,
                 int rank, int P, int method, int omp_threads = 1)
{
    std::priority_queue<std::pair<float, uint32_t>> local;
    if (method == 1) {
        local = (omp_threads >= 2)
              ? ivf_pq_search_omp (q, vecdim, local_topk, nprobe, rerank_k, omp_threads)
              : ivf_pq_search_simd(q, vecdim, local_topk, nprobe, rerank_k);
    } else {
        local = (omp_threads >= 2)
              ? ivf_pq2_search_omp (q, vecdim, local_topk, nprobe, rerank_k, omp_threads)
              : ivf_pq2_search_simd(q, vecdim, local_topk, nprobe, rerank_k);
    }
    return mpi_gather_merge(local, shard_ids, k, local_topk, rank, P);
}

// 释放 B 组索引 (method 1/2 各自的 PQ + 公共 IVF)
inline void mpi_ivfpq_free(int method) {
    if (method == 1) free_ivf_pq();
    else             free_ivf_pq_v2();
    free_ivf();
}
