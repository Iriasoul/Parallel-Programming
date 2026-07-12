// mpi_hetero.cuh
// S6：分布式+异构协同分流（MPI GPU-LSH ∥ CPU-IVF）
// rank 0 : GPU 节点 (LSH/exact）, 其余为 CPU 节点 IVF(AVX2+OpenMP)

// 启动： mpiexec -n <R> main.exe 6 [data_dir] [gpu_algo] [gpu_frac] [nprobe] [cpu_threads]
// gpu_algo=0 : 精确, 1 : LSH(默认)
#pragma once
#include "cpu_index.h"
#include "lsh_gpu.cuh"
#include "gpu_util.cuh"
#include <mpi.h>

inline void run_mpi_hetero(const float* base, size_t N_, const float* query, size_t M,
                           size_t d_, size_t k, const int* gt, size_t gtd,
                           int gpu_algo, double gpu_frac, int nprobe, int nthr) {
    MPI_Init(NULL, NULL);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int N = (int)N_, d = (int)d_, batch_B = 512, K_lsh = 256, rerank_lsh = 500;
    const double BASE = BENCH_BASELINE_US;
    const char* gpu_name = (gpu_algo == 0) ? "exact" : "LSH";

    // 切分：rank0=GPU 段 gpu_frac*M，其余均分给 size-1 个 CPU 节点
    std::vector<int> cnt(size, 0), displ(size, 0);
    size_t gpuN = (size == 1) ? M : (size_t)(gpu_frac * M + 0.5);
    if (gpuN > M) gpuN = M;
    cnt[0] = (int)gpuN;
    if (size > 1) {
        size_t rest = M - gpuN, per = rest / (size - 1), extra = rest % (size - 1);
        for (int r = 1; r < size; ++r) cnt[r] = (int)(per + (r <= (int)extra ? 1 : 0));
    }
    for (int r = 1; r < size; ++r) displ[r] = displ[r-1] + cnt[r-1];
    size_t lo = displ[rank], hi = lo + cnt[rank];
    std::vector<uint32_t> local((size_t)cnt[rank] * k, 0);

    // 准备
    cublasHandle_t h = nullptr; float* dB = nullptr;
    if (rank == 0) {
        CUBLAS_CHECK(cublasCreate(&h));
        if (gpu_algo == 0) { CUDA_CHECK(cudaMalloc(&dB, (size_t)N*d*sizeof(float)));
                             CUDA_CHECK(cudaMemcpy(dB, base, (size_t)N*d*sizeof(float), cudaMemcpyHostToDevice)); }
        else lsh_gpu_build(base, N, d, K_lsh, batch_B, h);
    } else {
        build_ivf(base, N, d, 256);
    }

    // rank 内检索
    std::vector<std::vector<uint32_t>> res(M);
    auto do_search = [&]() {
        if (cnt[rank] == 0) return;
        if (rank == 0) {
            if (gpu_algo == 0) {
                const int nt=256; size_t sh=(size_t)nt*(sizeof(float)+sizeof(int));
                const float alpha=1.f,beta=0.f;
                float *dQ,*dS; uint32_t* dIdx;
                CUDA_CHECK(cudaMalloc(&dQ,  (size_t)batch_B*d*sizeof(float)));
                CUDA_CHECK(cudaMalloc(&dS,  (size_t)batch_B*N*sizeof(float)));
                CUDA_CHECK(cudaMalloc(&dIdx,(size_t)batch_B*k*sizeof(uint32_t)));
                for (size_t off=0; off<(size_t)cnt[0]; off+=batch_B) {
                    int B=(int)std::min<size_t>(batch_B,cnt[0]-off);
                    CUDA_CHECK(cudaMemcpy(dQ,query+(lo+off)*d,(size_t)B*d*sizeof(float),cudaMemcpyHostToDevice));
                    CUBLAS_CHECK(cublasSgemm(h,CUBLAS_OP_T,CUBLAS_OP_N,N,B,d,&alpha,dB,d,dQ,d,&beta,dS,N));
                    size_t tot=(size_t)B*N; ip_to_dist<<<(int)((tot+255)/256),256>>>(dS,tot);
                    topk_rows<<<B,nt,sh>>>(dS,N,(int)k,dIdx,nullptr);
                    CUDA_CHECK(cudaMemcpy(local.data()+off*k,dIdx,(size_t)B*k*sizeof(uint32_t),cudaMemcpyDeviceToHost));
                }
                cudaFree(dQ);cudaFree(dS);cudaFree(dIdx);
            } else {
                lsh_gpu_search(query, k, rerank_lsh, lo, hi, res);
                for (int i=0;i<cnt[rank];++i)
                    for (size_t j=0;j<k&&j<res[lo+i].size();++j) local[(size_t)i*k+j]=res[lo+i][j];
            }
        } else {
            cpu_ivf_search_range(query, d, k, nprobe, lo, hi, res, nthr);
            for (int i=0;i<cnt[rank];++i)
                for (size_t j=0;j<k&&j<res[lo+i].size();++j) local[(size_t)i*k+j]=res[lo+i][j];
        }
    };

    do_search(); MPI_Barrier(MPI_COMM_WORLD);         // 预热
    double t0 = MPI_Wtime();
    do_search();                                       // 计时
    double t_local = MPI_Wtime() - t0;
    double t_max = 0; MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // 归并
    std::vector<int> rcnt(size), rdisp(size);
    for (int r=0;r<size;++r){ rcnt[r]=cnt[r]*(int)k; rdisp[r]=displ[r]*(int)k; }
    std::vector<uint32_t> all;
    if (rank==0) all.resize(M*k);
    MPI_Gatherv(local.data(),cnt[rank]*(int)k,MPI_UINT32_T,all.data(),rcnt.data(),rdisp.data(),MPI_UINT32_T,0,MPI_COMM_WORLD);
    std::vector<double> times(size,0);
    MPI_Gather(&t_local,1,MPI_DOUBLE,times.data(),1,MPI_DOUBLE,0,MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<std::vector<uint32_t>> full(M);
        for (size_t i=0;i<M;++i) full[i].assign(all.begin()+i*k,all.begin()+(i+1)*k);
        double rec=compute_recall(full,gt,gtd,k), us=t_max/M*1e6, qps=M/t_max;
        std::printf("\n== [S6] MPI + 异构分流 (MPI %d 进程: 1 GPU-%s + %d CPU-IVF) ==\n", size, gpu_name, size-1);
        std::printf("gpu_algo=%d gpu_frac=%.2f nprobe=%d cpu_threads=%d  baseline=%.1f us/q\n", gpu_algo, gpu_frac, nprobe, nthr, BASE);
        std::printf("recall=%.4f  makespan=%.1f ms  us/query=%.1f  qps=%.0f  speedup=%.2f\n", rec, t_max*1000.0, us, qps, BASE/us);
        std::printf("per-rank search time(ms): ");
        for (int r=0;r<size;++r) std::printf("[r%d %s %dq %.1fms] ", r, r==0?"GPU":"CPU", cnt[r], times[r]*1000.0);
        std::printf("\n====================\n");
    }

    if (rank==0) { if (gpu_algo==0) cudaFree(dB); else lsh_gpu_free(); cublasDestroy(h); }
    else free_ivf();
    MPI_Finalize();
}
