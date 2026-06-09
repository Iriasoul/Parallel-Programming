// ===========================================================================
//  lab4 MPI 编程 —— 在 lab2-SIMD / lab3-多线程 基础上做 MPI 多进程并行
//  策略表 (在线查询所用并行方式):
//   0  = Serial IVF baseline    单进程全量, 加速比基准 (建议 -np 1)
//   --- A 组: MPI-IVF ---
//   1  = MPI-IVF  block  + Gather归并            np=1 时应≈策略0
//   2  = MPI-IVF  cyclic + Gather归并            循环划分对比
//   3  = MPI-IVF  block  + 树形Reduce(自定义Op)  对比 Gather 的归并方式
//   4  = MPI-IVF  block  + OpenMP 混合 (P×T)     进程内簇级 OMP 动态调度
//   5  = MPI-IVF  批量通信                       全部 query 一次性 Gather, 摊薄通信
//   6  = MPI-IVF  非阻塞 overlap                 Igather, 计算与通信重叠 (流水线)
//   --- B 组: MPI-IVF-PQ ---
//   7  = MPI-IVF-PQ 方法一(全局PQ)    纯 MPI
//   8  = MPI-IVF-PQ 方法一(全局PQ)    + OpenMP
//   9  = MPI-IVF-PQ 方法二(簇内独立PQ) 纯 MPI
//   10 = MPI-IVF-PQ 方法二(簇内独立PQ) + OpenMP
//   --- C 组: 图索引 ---
//   11 = MPI-HNSW 多分块 (纯MPI)         每进程一张子图, searchKnn + Gather
//   12 = MPI-HNSW 多分块 + OpenMP        进程内再切 T 子图, OMP 多子图搜
//   13 = MPI IVF+HNSW 混合               每簇一张 HNSW, 簇按 c%P 分配, 簇间 MPI 并行
//   14 = HNSW-on-HNSW                    同 13 簇内 HNSW + 顶层路由 HNSW 选簇
//  切策略: 改下方 strategy 后重新 make. 进程数 P 由 mpiexec -np 决定.
// ===========================================================================

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <sys/time.h>
#include <queue>
#include <cstdint>
#include <mpi.h>

#include "ivf.h"
#include "mpi_common.h"
#include "mpi_ivf.h"
#include "mpi_ivf_pq.h"
#include "mpi_hnsw.h"

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if(!fin.is_open()) {
        std::cerr << "Failed to open " << data_path << std::endl;
        exit(-1);
    }
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(size_t i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();
    std::cerr<<"load data "<<data_path<<"\n";
    return data;
}

struct SearchResult {
    float recall;
    int64_t latency;
};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank = 0, P = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
    std::string data_path = "/anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    test_number = 2000;
    const size_t k = 10;
    std::vector<SearchResult> results(test_number);

    int strategy = 2;        // 0..14
    int nlist    = 256;      // IVF 簇数
    int nprobe   = 16;       // IVF 探查簇数 (IVF / IVF-PQ / IVF+HNSW 用)
    int rerank_k = 500;      // B 组 PQ 精排候选数
    int PQ_M     = 8;        // B 组 PQ 子空间数
    size_t hnsw_M = 16, hnsw_efc = 200; int hnsw_efs = 64;  // C 组 HNSW 参数
    size_t local_topk = k;   // 每 rank 上报候选数 (>=k 可换更高召回)
    int online_omp = 4;      // 策略4/8/10 线程数, 策略12 子图数 (P×T)
    PartitionScheme part = PART_BLOCK;

    // 策略相关的派生设置
    PartitionScheme build_part = (strategy == 2) ? PART_CYCLIC : part;
    int online_threads = (strategy == 4 || strategy == 8 || strategy == 10) ? online_omp : 1;
    int pq_method = (strategy == 7 || strategy == 8) ? 1
                  : (strategy == 9 || strategy == 10) ? 2 : 0;
    bool use_router = (strategy == 14);

    ivf_ns::nlist = nlist;
    if (rank == 0) std::cout << "Running Strategy: " << strategy << "  P=" << P << std::endl;

    // 离线索引构建 (不计入查询延迟)
    std::vector<uint32_t> shard_ids;
    if (strategy == 0) {
        if (rank == 0) build_ivf(base, base_number, vecdim, nlist);
    } else if (strategy >= 1 && strategy <= 6) {
        mpi_ivf_build_local(base, base_number, vecdim, nlist, build_part, rank, P, shard_ids);
    } else if (strategy >= 7 && strategy <= 10) {
        mpi_ivfpq_build_local(base, base_number, vecdim, nlist, PQ_M, pq_method,
                              build_part, rank, P, shard_ids);
    } else if (strategy == 11 || strategy == 12) {
        int n_sub = (strategy == 12) ? online_omp : 1;
        mpi_hnsw_build_local(base, base_number, vecdim, build_part, rank, P,
                             n_sub, hnsw_M, hnsw_efc, hnsw_efs, shard_ids);
    } else if (strategy == 13 || strategy == 14) {
        mpi_ivfhnsw_build(base, base_number, vecdim, nlist, rank, P,
                          hnsw_M, hnsw_efc, hnsw_efs, use_router);
    }

    float avg_recall = 0, avg_latency = 0;

    if (strategy == 5) {
        // 批量通信: 独占整个查询流程
        mpi_ivf_batch(test_query, test_number, vecdim, k, nprobe, local_topk,
                      test_gt, test_gt_d, shard_ids, rank, P, avg_recall, avg_latency);
    }
    else if (strategy == 6) {
        // 非阻塞 overlap: 独占整个查询流程
        mpi_ivf_pipeline(test_query, test_number, vecdim, k, nprobe, local_topk,
                         test_gt, test_gt_d, shard_ids, rank, P, avg_recall, avg_latency);
    }
    else {
        // 逐查询风格 (策略 0,1,2,3,4)
        for(size_t i = 0; i < test_number; ++i) {
            struct timeval start, end;
            gettimeofday(&start, NULL);

            std::priority_queue<std::pair<float, uint32_t>> res;
            float* q = test_query + i*vecdim;

            if (strategy == 0) {
                if (rank == 0) res = ivf_search_simd(q, vecdim, k, nprobe);
            }
            else if (strategy == 1 || strategy == 2 || strategy == 4) {
                res = mpi_ivf_gather(q, vecdim, k, nprobe, local_topk, shard_ids, rank, P, online_threads);
            }
            else if (strategy == 3) {
                res = mpi_ivf_reduce(q, vecdim, k, nprobe, local_topk, shard_ids, rank, P);
            }
            else if (strategy >= 7 && strategy <= 10) {
                res = mpi_ivfpq_gather(q, vecdim, k, nprobe, rerank_k, local_topk,
                                       shard_ids, rank, P, pq_method, online_threads);
            }
            else if (strategy == 11) {
                res = mpi_hnsw_gather(q, vecdim, k, local_topk, shard_ids, rank, P);
            }
            else if (strategy == 12) {
                res = mpi_hnsw_gather_sub(q, vecdim, k, local_topk, shard_ids, rank, P, online_omp);
            }
            else if (strategy == 13 || strategy == 14) {
                res = mpi_ivfhnsw_search(q, vecdim, k, nprobe, local_topk, rank, P, use_router);
            }

            gettimeofday(&end, NULL);
            int64_t diff = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);

            float recall = 0.0f;
            if (rank == 0) {
                std::set<uint32_t> gtset;
                for(size_t j = 0; j < k; ++j) gtset.insert(test_gt[j + i*test_gt_d]);
                size_t acc = 0;
                while (!res.empty()) {
                    if(gtset.count(res.top().second)) ++acc;
                    res.pop();
                }
                recall = (float)acc/k;
            }
            results[i] = {recall, diff};
        }

        // 延迟收口到 root: 每条 query 取各进程耗时最大值, 再平均
        std::vector<int64_t> lat(test_number), latmax(test_number);
        for(size_t i = 0; i < test_number; ++i) lat[i] = results[i].latency;
        MPI_Reduce(lat.data(), latmax.data(), (int)test_number,
                   MPI_INT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            for(size_t i = 0; i < test_number; ++i) {
                avg_recall  += results[i].recall;
                avg_latency += latmax[i];
            }
            avg_recall  /= test_number;
            avg_latency /= test_number;
        }
    }

    if (rank == 0) {
        std::cout << "Avg Recall: " << avg_recall
                  << "\nAvg Latency: " << avg_latency << " us\n";
        std::cout << "nlist = " << nlist << ", nprobe = " << nprobe
                  << ", part = " << (build_part==PART_BLOCK?"block":"cyclic")
                  << ", local_topk = " << local_topk;
        if (strategy == 4 || strategy == 8 || strategy == 10) std::cout << ", omp_threads = " << online_omp;
        if (strategy >= 7 && strategy <= 10)
            std::cout << ", rerank_k = " << rerank_k << ", PQ_M = " << PQ_M
                      << ", method = " << pq_method;
        if (strategy >= 11 && strategy <= 14) std::cout << ", ef_search = " << hnsw_efs;
        if (strategy == 12) std::cout << ", n_sub = " << online_omp;
        std::cout << "\n\n" << "====================" << "\n";
    }

    delete[] test_query; delete[] test_gt; delete[] base;
    if      (strategy >= 0  && strategy <= 6)  free_ivf();
    else if (strategy >= 7  && strategy <= 10) mpi_ivfpq_free(pq_method);
    else if (strategy == 11)                   free_hnsw();
    else if (strategy == 12)                   free_hnsw_sub();
    else if (strategy == 13 || strategy == 14) mpi_ivfhnsw_free();

    MPI_Finalize();
    return 0;
}
