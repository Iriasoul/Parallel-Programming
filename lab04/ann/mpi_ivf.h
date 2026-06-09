#pragma once

#include <mpi.h>
#include <vector>
#include <queue>
#include <cstdint>
#include <cstring>
#include <set>
#include <algorithm>
#include <sys/time.h>
#include <omp.h>

#include "ivf.h"
#include "mpi_common.h"

// 离线: 在本 rank 的数据分片上构建 IVF 索引, 输出该分片的全局 id 映射表 shard_ids
inline void
mpi_ivf_build_local(const float* full_base, size_t N, size_t vecdim,
                    int nlist, PartitionScheme part, int rank, int P,
                    std::vector<uint32_t>& shard_ids /*out*/)
{
    omp_set_num_threads(1);   // 同节点多 rank 并存, 建索引也单线程避免过度订阅
    shard_ids = mpi_compute_shard_ids(N, rank, P, part);
    float* local_base = mpi_materialize_local_base(full_base, vecdim, shard_ids);
    build_ivf(local_base, shard_ids.size(), vecdim, nlist);
    delete[] local_base;
}

// 本地搜索: omp_threads>=2 则 OpenMP 簇级并行 (策略4), 否则单线程 SIMD
static inline std::priority_queue<std::pair<float, uint32_t>>
mpi_ivf_local_search(float* q, size_t vecdim, size_t local_topk, int nprobe, int omp_threads)
{
    if (omp_threads >= 2)
        return ivf_search_omp(q, vecdim, local_topk, nprobe, omp_threads);
    return ivf_search_simd(q, vecdim, local_topk, nprobe);
}

// 策略 1/2/4: 单查询 本地搜索 + 两次 Gather 到 root + 归并 top-k 
inline std::priority_queue<std::pair<float, uint32_t>>
mpi_ivf_gather(float* q, size_t vecdim, size_t k, int nprobe, size_t local_topk,
               const std::vector<uint32_t>& shard_ids, int rank, int P,
               int omp_threads = 1)
{
    auto local = mpi_ivf_local_search(q, vecdim, local_topk, nprobe, omp_threads);
    return mpi_gather_merge(local, shard_ids, k, local_topk, rank, P);
}

// 策略 3: 单查询 本地搜索 + 树形 Reduce 归并 (自定义算子) 
inline std::priority_queue<std::pair<float, uint32_t>>
mpi_ivf_reduce(float* q, size_t vecdim, size_t k, int nprobe, size_t local_topk,
               const std::vector<uint32_t>& shard_ids, int rank, int /*P*/)
{
    auto local = ivf_search_simd(q, vecdim, local_topk, nprobe);

    std::vector<float> sd; std::vector<uint32_t> sid;
    mpi_flatten_local(local, shard_ids, local_topk, sd, sid);

    std::vector<MpiCand> send(local_topk), recv(local_topk);
    for (size_t i = 0; i < local_topk; ++i) { send[i].d = sd[i]; send[i].id = sid[i]; }

    MPI_Datatype type; MPI_Op op;
    g_topk_K = (int)local_topk;                 // 告诉算子每元素含多少候选
    mpi_topk_get_type_op((int)local_topk, type, op);
    MPI_Reduce(send.data(), recv.data(), 1, type, op, 0, MPI_COMM_WORLD);

    std::priority_queue<std::pair<float, uint32_t>> res;
    if (rank == 0) {
        for (size_t i = 0; i < local_topk; ++i) {
            if (recv[i].id == UINT32_MAX) continue;
            if (res.size() < k || recv[i].d < res.top().first) {
                res.push({recv[i].d, recv[i].id});
                if (res.size() > k) res.pop();
            }
        }
    }
    return res;
}

// 策略 5: 批量通信
inline void
mpi_ivf_batch(float* test_query, size_t test_number, size_t vecdim,
              size_t k, int nprobe, size_t local_topk,
              const int* test_gt, size_t test_gt_d,
              const std::vector<uint32_t>& shard_ids, int rank, int P,
              float& avg_recall, float& avg_latency)
{
    size_t per = test_number * local_topk;          // 本 rank 全部 query 的候选数
    std::vector<float>    sd_all(per);
    std::vector<uint32_t> sid_all(per);
    std::vector<float>    rd_all;
    std::vector<uint32_t> rid_all;
    if (rank == 0) { rd_all.resize(per * (size_t)P); rid_all.resize(per * (size_t)P); }

    std::vector<uint32_t> res_ids(test_number * k, UINT32_MAX);  // 归并出的 top-k id

    MPI_Barrier(MPI_COMM_WORLD);
    struct timeval t0, t1; gettimeofday(&t0, NULL);

    // 本地搜索全部 query 
    for (size_t qi = 0; qi < test_number; ++qi) {
        auto local = ivf_search_simd(test_query + qi * vecdim, vecdim, local_topk, nprobe);
        std::vector<float> sd; std::vector<uint32_t> sid;
        mpi_flatten_local(local, shard_ids, local_topk, sd, sid);
        std::copy(sd.begin(),  sd.end(),  sd_all.begin()  + qi * local_topk);
        std::copy(sid.begin(), sid.end(), sid_all.begin() + qi * local_topk);
    }

    // 一次性 Gather
    MPI_Gather(sd_all.data(),  (int)per, MPI_FLOAT,
               rank == 0 ? rd_all.data()  : nullptr, (int)per, MPI_FLOAT,    0, MPI_COMM_WORLD);
    MPI_Gather(sid_all.data(), (int)per, MPI_UINT32_T,
               rank == 0 ? rid_all.data() : nullptr, (int)per, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // root 逐 query 归并
    if (rank == 0) {
        std::vector<float>    md(local_topk * (size_t)P);
        std::vector<uint32_t> mid(local_topk * (size_t)P);
        for (size_t qi = 0; qi < test_number; ++qi) {
            for (int r = 0; r < P; ++r) {
                size_t src = (size_t)r * per + qi * local_topk;
                std::copy(rd_all.begin()  + src, rd_all.begin()  + src + local_topk, md.begin()  + (size_t)r * local_topk);
                std::copy(rid_all.begin() + src, rid_all.begin() + src + local_topk, mid.begin() + (size_t)r * local_topk);
            }
            auto res = mpi_merge_topk(md, mid, k);
            size_t idx = 0;
            while (!res.empty()) { res_ids[qi * k + idx] = res.top().second; res.pop(); ++idx; }
        }
    }

    gettimeofday(&t1, NULL);
    double total_us = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_usec - t0.tv_usec);

    // 召回 
    if (rank == 0) {
        double recall_sum = 0.0;
        for (size_t qi = 0; qi < test_number; ++qi) {
            std::set<uint32_t> gtset;
            for (size_t j = 0; j < k; ++j) gtset.insert((uint32_t)test_gt[j + qi * test_gt_d]);
            size_t acc = 0;
            for (size_t j = 0; j < k; ++j) {
                uint32_t id = res_ids[qi * k + j];
                if (id != UINT32_MAX && gtset.count(id)) ++acc;
            }
            recall_sum += (double)acc / k;
        }
        avg_recall  = (float)(recall_sum / test_number);
        avg_latency = (float)(total_us / test_number);
    }
}

// 策略 6: 非阻塞 overlap 
inline void
mpi_ivf_pipeline(float* test_query, size_t test_number, size_t vecdim,
                 size_t k, int nprobe, size_t local_topk,
                 const int* test_gt, size_t test_gt_d,
                 const std::vector<uint32_t>& shard_ids, int rank, int P,
                 float& avg_recall, float& avg_latency)
{
    // 双缓冲
    std::vector<float>    sd[2]  = { std::vector<float>(local_topk),  std::vector<float>(local_topk) };
    std::vector<uint32_t> sid[2] = { std::vector<uint32_t>(local_topk), std::vector<uint32_t>(local_topk) };
    std::vector<float>    rd[2];
    std::vector<uint32_t> rid[2];
    if (rank == 0)
        for (int b = 0; b < 2; ++b) { rd[b].resize(local_topk * (size_t)P); rid[b].resize(local_topk * (size_t)P); }

    MPI_Request req_d[2]  = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
    MPI_Request req_id[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
    std::vector<uint32_t> res_ids(test_number * k, UINT32_MAX);

    auto search_fill = [&](size_t qi, int buf) {
        auto local = ivf_search_simd(test_query + qi * vecdim, vecdim, local_topk, nprobe);
        std::vector<float> td; std::vector<uint32_t> tid;
        mpi_flatten_local(local, shard_ids, local_topk, td, tid);
        std::copy(td.begin(),  td.end(),  sd[buf].begin());
        std::copy(tid.begin(), tid.end(), sid[buf].begin());
    };
    auto post = [&](int buf) {
        MPI_Igather(sd[buf].data(),  (int)local_topk, MPI_FLOAT,
                    rank == 0 ? rd[buf].data()  : nullptr, (int)local_topk, MPI_FLOAT,    0, MPI_COMM_WORLD, &req_d[buf]);
        MPI_Igather(sid[buf].data(), (int)local_topk, MPI_UINT32_T,
                    rank == 0 ? rid[buf].data() : nullptr, (int)local_topk, MPI_UINT32_T, 0, MPI_COMM_WORLD, &req_id[buf]);
    };
    auto finish = [&](size_t qi, int buf) {
        MPI_Wait(&req_d[buf],  MPI_STATUS_IGNORE);
        MPI_Wait(&req_id[buf], MPI_STATUS_IGNORE);
        if (rank == 0) {
            auto res = mpi_merge_topk(rd[buf], rid[buf], k);  // (dist[], id[])
            size_t idx = 0;
            while (!res.empty()) { res_ids[qi * k + idx] = res.top().second; res.pop(); ++idx; }
        }
    };

    MPI_Barrier(MPI_COMM_WORLD);
    struct timeval t0, t1; gettimeofday(&t0, NULL);

    if (test_number > 0) {
        search_fill(0, 0);
        post(0);
        for (size_t qi = 1; qi < test_number; ++qi) {
            int cur = (int)(qi & 1), prev = (int)((qi - 1) & 1);
            search_fill(qi, cur);     // 本条计算 (与上条 Igather 重叠)
            finish(qi - 1, prev);     // 等上条通信完成 + 归并
            post(cur);                // 发起本条通信
        }
        finish(test_number - 1, (int)((test_number - 1) & 1));
    }

    gettimeofday(&t1, NULL);
    double total_us = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_usec - t0.tv_usec);

    if (rank == 0) {
        double recall_sum = 0.0;
        for (size_t qi = 0; qi < test_number; ++qi) {
            std::set<uint32_t> gtset;
            for (size_t j = 0; j < k; ++j) gtset.insert((uint32_t)test_gt[j + qi * test_gt_d]);
            size_t acc = 0;
            for (size_t j = 0; j < k; ++j) {
                uint32_t id = res_ids[qi * k + j];
                if (id != UINT32_MAX && gtset.count(id)) ++acc;
            }
            recall_sum += (double)acc / k;
        }
        avg_recall  = (float)(recall_sum / test_number);
        avg_latency = (float)(total_us / test_number);
    }
}
