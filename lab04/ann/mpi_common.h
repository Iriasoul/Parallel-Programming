#pragma once

#include <mpi.h>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <queue>
#include <utility>
#include <cfloat>
#include <set>
#include <algorithm>

// 任务划分方式
enum PartitionScheme { PART_BLOCK = 0, PART_CYCLIC = 1 };

// 计算本 rank 全局向量 id列表
//   block : 连续分块, 前 (N % P) 个 rank 各多分 1 个, 实现负载均衡
//   cyclic: 循环划分, 全局 id i 分给 rank (i % P)
inline std::vector<uint32_t>
mpi_compute_shard_ids(size_t N, int rank, int P, PartitionScheme part)
{
    std::vector<uint32_t> ids;
    if (part == PART_BLOCK) {
        size_t q = N / (size_t)P, rem = N % (size_t)P;
        size_t start = (size_t)rank * q + std::min((size_t)rank, rem);
        size_t cnt   = q + ((size_t)rank < rem ? 1 : 0);
        ids.reserve(cnt);
        for (size_t i = 0; i < cnt; ++i) ids.push_back((uint32_t)(start + i));
    } else { // PART_CYCLIC
        for (size_t i = (size_t)rank; i < N; i += (size_t)P)
            ids.push_back((uint32_t)i);
    }
    return ids;
}

// 把本 rank 负责的全局向量收集为连续的本地 base 缓冲 (调用方负责 delete[])
inline float*
mpi_materialize_local_base(const float* full_base, size_t vecdim,
                           const std::vector<uint32_t>& shard_ids)
{
    size_t n = shard_ids.size();
    float* local_base = new float[n * vecdim];
    for (size_t i = 0; i < n; ++i)
        std::memcpy(local_base + i * vecdim,
                    full_base + (size_t)shard_ids[i] * vecdim,
                    vecdim * sizeof(float));
    return local_base;
}

// 把局部 top-k 优先队列拍平成定长 (dist,id) 数组, 不足用 (FLT_MAX, UINT32_MAX) 填充;
// 同时把局部下标 (t.second) 通过 shard_ids 映射回全局原始 id
inline void
mpi_flatten_local(std::priority_queue<std::pair<float, uint32_t>>& local,
                  const std::vector<uint32_t>& shard_ids,
                  size_t local_topk,
                  std::vector<float>& out_d, std::vector<uint32_t>& out_id)
{
    out_d.assign(local_topk, FLT_MAX);
    out_id.assign(local_topk, UINT32_MAX);
    size_t idx = 0;
    while (!local.empty() && idx < local_topk) {
        auto t = local.top(); local.pop();
        out_d[idx]  = t.first;
        out_id[idx] = (t.second < shard_ids.size()) ? shard_ids[t.second] : UINT32_MAX;
        ++idx;
    }
}

// root 端: 把收集到的若干候选 (dist,id) 归并为全局 top-k
inline std::priority_queue<std::pair<float, uint32_t>>
mpi_merge_topk(const std::vector<float>& dists,
               const std::vector<uint32_t>& ids, size_t k)
{
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (size_t i = 0; i < dists.size(); ++i) {
        if (ids[i] == UINT32_MAX) continue;
        float d = dists[i];
        if (q.size() < k || d < q.top().first) {
            q.push({d, ids[i]});
            if (q.size() > k) q.pop();
        }
    }
    return q;
}

// 单次查询召回
inline double
mpi_recall_one(std::priority_queue<std::pair<float, uint32_t>> res,
               const int* gt_row, size_t k)
{
    std::set<uint32_t> gtset;
    for (size_t j = 0; j < k; ++j) gtset.insert((uint32_t)gt_row[j]);
    size_t acc = 0;
    while (!res.empty()) {
        if (gtset.count(res.top().second)) ++acc;
        res.pop();
    }
    return (double)acc / (double)k;
}

// 归并: 给定本 rank 的局部 top-k 队列 - 映射全局 id - 两次 Gather 到 root - 归并 top-k
inline std::priority_queue<std::pair<float, uint32_t>>
mpi_gather_merge(std::priority_queue<std::pair<float, uint32_t>>& local,
                 const std::vector<uint32_t>& shard_ids,
                 size_t k, size_t local_topk, int rank, int P)
{
    std::vector<float>    sd;  std::vector<uint32_t> sid;
    mpi_flatten_local(local, shard_ids, local_topk, sd, sid);

    std::vector<float>    rd;  std::vector<uint32_t> rid;
    if (rank == 0) { rd.resize(local_topk * (size_t)P); rid.resize(local_topk * (size_t)P); }

    MPI_Gather(sd.data(),  (int)local_topk, MPI_FLOAT,
               rank == 0 ? rd.data()  : nullptr, (int)local_topk, MPI_FLOAT,    0, MPI_COMM_WORLD);
    MPI_Gather(sid.data(), (int)local_topk, MPI_UINT32_T,
               rank == 0 ? rid.data() : nullptr, (int)local_topk, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    std::priority_queue<std::pair<float, uint32_t>> res;
    if (rank == 0) res = mpi_merge_topk(rd, rid, k);
    return res;
}

// 同上, 但 local 队列里的 id 已经是全局原始 id, 可不经 shard_ids 映射
inline std::priority_queue<std::pair<float, uint32_t>>
mpi_gather_merge_global(std::priority_queue<std::pair<float, uint32_t>>& local,
                        size_t k, size_t local_topk, int rank, int P)
{
    std::vector<float>    sd(local_topk, FLT_MAX);
    std::vector<uint32_t> sid(local_topk, UINT32_MAX);
    size_t idx = 0;
    while (!local.empty() && idx < local_topk) {
        sd[idx] = local.top().first; sid[idx] = local.top().second; local.pop(); ++idx;
    }
    std::vector<float>    rd;  std::vector<uint32_t> rid;
    if (rank == 0) { rd.resize(local_topk * (size_t)P); rid.resize(local_topk * (size_t)P); }

    MPI_Gather(sd.data(),  (int)local_topk, MPI_FLOAT,
               rank == 0 ? rd.data()  : nullptr, (int)local_topk, MPI_FLOAT,    0, MPI_COMM_WORLD);
    MPI_Gather(sid.data(), (int)local_topk, MPI_UINT32_T,
               rank == 0 ? rid.data() : nullptr, (int)local_topk, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    std::priority_queue<std::pair<float, uint32_t>> res;
    if (rank == 0) res = mpi_merge_topk(rd, rid, k);
    return res;
}

// 树形 Reduce 归并

struct MpiCand { float d; uint32_t id; };   // 8 字节, 无填充

// 规约算子读取它得知每个元素含多少个 MpiCand (调用 reduce 前需先设置)
static int g_topk_K = 0;

// 自定义规约: inout = merge(in, inout), 保留距离最小的 K 个
inline void mpi_topk_merge_op(void* inv, void* inoutv, int* len, MPI_Datatype*) {
    MpiCand* in = (MpiCand*)inv;
    MpiCand* io = (MpiCand*)inoutv;
    int K = g_topk_K;
    for (int l = 0; l < *len; ++l) {
        MpiCand* a = in + (size_t)l * K;   // 来自另一进程
        MpiCand* b = io + (size_t)l * K;   // 本地累积, 就地更新
        std::priority_queue<std::pair<float, uint32_t>> q;   // max-heap
        for (int i = 0; i < K; ++i)
            if (b[i].id != UINT32_MAX) { q.push({b[i].d, b[i].id}); if ((int)q.size() > K) q.pop(); }
        for (int i = 0; i < K; ++i)
            if (a[i].id != UINT32_MAX) {
                if ((int)q.size() < K || a[i].d < q.top().first) {
                    q.push({a[i].d, a[i].id});
                    if ((int)q.size() > K) q.pop();
                }
            }
        int idx = 0;
        while (!q.empty()) { b[idx].d = q.top().first; b[idx].id = q.top().second; q.pop(); ++idx; }
        for (; idx < K; ++idx) { b[idx].d = FLT_MAX; b[idx].id = UINT32_MAX; }
    }
}

// 懒创建并缓存 K 个 MpiCand 连续的数据类型与规约算子 (K 变化时重建)
inline void mpi_topk_get_type_op(int K, MPI_Datatype& type, MPI_Op& op) {
    static MPI_Datatype cached_type = MPI_DATATYPE_NULL;
    static MPI_Op       cached_op   = MPI_OP_NULL;
    static int          cached_K    = -1;
    if (cached_K != K) {
        if (cached_type != MPI_DATATYPE_NULL) MPI_Type_free(&cached_type);
        if (cached_op   != MPI_OP_NULL)       MPI_Op_free(&cached_op);

        MPI_Datatype cand, cand_resized;
        int          blk[2]   = {1, 1};
        MPI_Aint     disp[2]  = {offsetof(MpiCand, d), offsetof(MpiCand, id)};
        MPI_Datatype types[2] = {MPI_FLOAT, MPI_UINT32_T};
        MPI_Type_create_struct(2, blk, disp, types, &cand);
        MPI_Type_create_resized(cand, 0, sizeof(MpiCand), &cand_resized);
        MPI_Type_contiguous(K, cand_resized, &cached_type);
        MPI_Type_commit(&cached_type);
        MPI_Op_create(&mpi_topk_merge_op, 1 /*commute*/, &cached_op);
        MPI_Type_free(&cand);
        MPI_Type_free(&cand_resized);
        cached_K = K;
    }
    type = cached_type;
    op   = cached_op;
}
