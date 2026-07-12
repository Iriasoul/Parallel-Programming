/*
  strategy 表：
    S0 = Flat暴力算法  
    S1 = CPU IVF / IVF-PQ 
    S2 = GPU cuBLAS     
    S3 = LSH - CPU 
    S4 = LSH - GPU   
    S5 = 异构协同分流 
    S6 = MPI + 异构分流  
*/
#include "bench_common.h"
#include "flat.h"
#include "cpu_search.h"
#include "gpu_util.cuh"
#include "gpu_coarse.cuh"
#include "lsh_cpu.h"
#include "lsh_gpu.cuh"
#include "hetero_split.cuh"
#include "mpi_hetero.cuh"

int main(int argc, char** argv) {
    int strategy       = (argc > 1) ? atoi(argv[1]) : 5;
    std::string dir    = (argc > 2) ? argv[2] : "../lab05/ann/data/";
    if (!dir.empty() && dir.back() != '/' && dir.back() != '\\') dir += "/";

    size_t nq, dq, ngt, dgt, nb, db;
    float* query = LoadData<float>(dir + ANN_QUERY_FILE, nq, dq);
    int*   gt    = LoadData<int>  (dir + ANN_GT_FILE,    ngt, dgt);
    float* base  = LoadData<float>(dir + ANN_BASE_FILE,  nb, db);
    const size_t d = db, k = 10, M = std::min<size_t>(2000, std::min(nq, ngt));

    int dev = 0; cudaDeviceProp p;
    if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess)
        std::printf("GPU: %s (%.1f GB)\n", p.name, p.totalGlobalMem / 1073741824.0);
    std::printf("strategy=%d  data=%s\n", strategy, dir.c_str());

    switch (strategy) {
        // 基线与单算法
        case 0: run_flat(base, nb, query, M, d, k, gt, dgt); break;
        case 1: {
            int threads = (argc > 3) ? atoi(argv[3]) : 8;
            run_cpu_ivf(base, nb, query, M, d, k, gt, dgt, threads); break;
        }
        case 2: {
            int batch_B = (argc > 3) ? atoi(argv[3]) : 512;
            run_gpu_coarse(base, nb, query, M, d, k, gt, dgt, batch_B); break;
        }
        case 3: {
            int threads = (argc > 3) ? atoi(argv[3]) : 8;
            run_lsh_cpu(base, nb, query, M, d, k, gt, dgt, threads); break;
        }
        case 4: {
            int batch_B = (argc > 3) ? atoi(argv[3]) : 512;
            run_lsh_gpu(base, nb, query, M, d, k, gt, dgt, batch_B); break;
        }
        // 融合算法
        case 5: {
            int batch_B    = (argc > 3) ? atoi(argv[3]) : 512;
            int threads    = (argc > 4) ? atoi(argv[4]) : 8;
            int nprobe     = (argc > 5) ? atoi(argv[5]) : 16;
            int gpu_algo   = (argc > 6) ? atoi(argv[6]) : 1;
            int rerank_lsh = (argc > 7) ? atoi(argv[7]) : 500;
            run_hetero_split(base, nb, query, M, d, k, gt, dgt, batch_B, threads, nprobe, gpu_algo, 256, rerank_lsh);
            break;
        }
        case 6: {
            int gpu_algo   = (argc > 3) ? atoi(argv[3]) : 1;
            double gpu_frac = (argc > 4) ? atof(argv[4]) : 0.5;
            int nprobe      = (argc > 5) ? atoi(argv[5]) : 16;
            int threads     = (argc > 6) ? atoi(argv[6]) : 4;
            run_mpi_hetero(base, nb, query, M, d, k, gt, dgt, gpu_algo, gpu_frac, nprobe, threads); break;
        }
        default: std::printf("unknown strategy %d (expect 0..6)\n", strategy);
    }

    delete[] query; delete[] gt; delete[] base;
    return 0;
}
