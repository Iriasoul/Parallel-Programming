// ===========================================================================
//  r_main.cc —— 批量测试用主程序 (main.cc 的命令行参数化版本)
//  逻辑与 main.cc 完全一致, 仅:
//    1) 所有参数从命令行读取 (脚本扫参数时无需重编)
//    2) rank0 额外输出一行机器可读的 "RESULT key=value ..." 供脚本解析
//  用法示例:
//    mpiexec -np 4 ./r_main --strategy 5 --nprobe 16 --nlist 256
//    mpiexec -np 2 ./r_main --strategy 4 --omp 4
//    mpiexec -np 4 ./r_main --strategy 9 --rerank_k 500 --pqm 8
//    mpiexec -np 8 ./r_main --strategy 13 --ef 64 --nprobe 16
//  参数: --strategy --nprobe --nlist --rerank_k --pqm --ef --omp
//        --part(block|cyclic) --topk(local_topk) --test_number
// ===========================================================================

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <sys/time.h>
#include <queue>
#include <cstdint>
#include <cstring>
#include <cstdlib>
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

// 简单命令行解析: --key value
static int    arg_int (int argc, char** argv, const char* key, int def) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], key)) return atoi(argv[i+1]);
    return def;
}
static std::string arg_str(int argc, char** argv, const char* key, const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], key)) return std::string(argv[i+1]);
    return def;
}

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

    const size_t k = 10;

    // ---- 命令行参数 ----
    int    strategy   = arg_int(argc, argv, "--strategy", 1);
    int    nlist      = arg_int(argc, argv, "--nlist",    256);
    int    nprobe     = arg_int(argc, argv, "--nprobe",   16);
    int    rerank_k   = arg_int(argc, argv, "--rerank_k", 500);
    int    PQ_M       = arg_int(argc, argv, "--pqm",      8);
    int    hnsw_efs   = arg_int(argc, argv, "--ef",       64);
    int    online_omp = arg_int(argc, argv, "--omp",      4);
    size_t local_topk = (size_t)arg_int(argc, argv, "--topk", (int)k);
    size_t tn         = (size_t)arg_int(argc, argv, "--test_number", 2000);
    std::string parts = arg_str(argc, argv, "--part", "block");

    size_t hnsw_M = 16, hnsw_efc = 200;
    test_number = (tn < test_number) ? tn : test_number;
    if (test_number == 0) test_number = 2000;
    std::vector<SearchResult> results(test_number);

    PartitionScheme part = (parts == "cyclic") ? PART_CYCLIC : PART_BLOCK;
    PartitionScheme build_part = (strategy == 2) ? PART_CYCLIC : part;
    int online_threads = (strategy == 4 || strategy == 8 || strategy == 10) ? online_omp : 1;
    int pq_method = (strategy == 7 || strategy == 8) ? 1
                  : (strategy == 9 || strategy == 10) ? 2 : 0;
    bool use_router = (strategy == 14);

    ivf_ns::nlist = nlist;

    // 离线索引构建
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
        mpi_ivf_batch(test_query, test_number, vecdim, k, nprobe, local_topk,
                      test_gt, test_gt_d, shard_ids, rank, P, avg_recall, avg_latency);
    }
    else if (strategy == 6) {
        mpi_ivf_pipeline(test_query, test_number, vecdim, k, nprobe, local_topk,
                         test_gt, test_gt_d, shard_ids, rank, P, avg_recall, avg_latency);
    }
    else {
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

    // 机器可读输出 (供脚本解析); 仅 rank0
    if (rank == 0) {
        double qps = (avg_latency > 0) ? 1e6 / avg_latency : 0;
        std::cout << "RESULT"
                  << " strategy=" << strategy
                  << " P=" << P
                  << " omp=" << online_omp
                  << " part=" << (build_part==PART_BLOCK?"block":"cyclic")
                  << " nprobe=" << nprobe
                  << " nlist=" << nlist
                  << " rerank_k=" << rerank_k
                  << " pqm=" << PQ_M
                  << " ef=" << hnsw_efs
                  << " topk=" << local_topk
                  << " test_number=" << test_number
                  << " recall=" << avg_recall
                  << " latency_us=" << avg_latency
                  << " qps=" << qps
                  << "\n";
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
