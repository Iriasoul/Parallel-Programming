//  Lab5 GPU 编程  ANN 选题
/*

暴力检索 :
    S0 = Flat : GPU 矩阵乘(cuBLAS) + CPU 选 top-k          (GPU+CPU)
    S1 = Flat : GPU 矩阵乘(cuBLAS) + GPU 选 top-k          (GPU+GPU)
IVF :
    S2 = IVF (group 切换分组策略: 0 顺序 / 1 按主簇 / 2 按完整探查簇集合 / 3 query-KMeans)
IVF-PQ :
    S3 = IVF-PQ 全局PQ + 重排
    S4 = IVF-PQ 簇内PQ + 重排

*/

#include "common.cuh"
#include "gpu_flat.cuh"
#include "gpu_ivf.cuh"
#include "gpu_ivf_pq.cuh"

int main() {
    const std::string dir = ANN_DATA_DIR;
    size_t M_all=0, N=0, gtr=0, gtd=0, dq=0, db=0;
    float* query = LoadData<float>(dir + ANN_QUERY_FILE, M_all, dq);
    int*   gt    = LoadData<int>  (dir + ANN_GT_FILE,    gtr, gtd);
    float* base  = LoadData<float>(dir + ANN_BASE_FILE,  N, db);
    const size_t d = db;

    // 配置
    int strategy = 2;      // 0 - 4  见上方策略表
    int nlist = 64; 
    int nprobe = 16;
    int group_mode = 1;      // 分组: 0顺序 / 1按主簇 / 2按完整簇集 / 3 query-KMeans
    size_t group_B = 512;    // GPU 每组查询数
    int rerank_k = 100;
    int PQ_M = 8; 
    int PQ_KSUB = 256;    // PQ 每段子码字数
    size_t test_number = 2000;   // 测试 query 数
    const size_t k = 10;
    const int R = 3;      // GPU 重复取平均


    size_t M = std::min(test_number, std::min(M_all, gtr));
    if (d % PQ_M != 0) { std::printf("ERROR: d=%d 不能被 PQ_M=%d 整除\n",(int)d,PQ_M); return 1; }

    bool is_flat        = (strategy<=1);
    bool need_global_pq = (strategy==3);
    bool need_pc_pq     = (strategy==4);
    int  fine_mode      = (strategy==2)?0 : (strategy==3)?1 : (strategy==4)?2 : -1;
    const char* sname[5] = {"Flat GPU+CPU","Flat GPU+GPU","IVF exact",
                            "IVF-PQ global + rerank","IVF-PQ per-cluster + rerank"};
    const char* gname[4] = {"sequential","by-primary","by-probeset","query-kmeans"};

    { int dev=0; cudaDeviceProp p; CUDA_CHECK(cudaGetDevice(&dev));
      CUDA_CHECK(cudaGetDeviceProperties(&p,dev));
      std::printf("\nGPU: %s (%.1f GB)\n", p.name, p.totalGlobalMem/1073741824.0); }
    std::printf("Running Strategy: %d  (%s)\n", strategy, sname[strategy]);

    std::vector<std::vector<uint32_t>> res;          // GPU 结果
    std::vector<std::vector<uint32_t>> res_cpu;      // CPU 参考
    double recall=0, gpu_ms=0, overlap=0, cpu_recall=0, cpu_ms=0;

    if (is_flat) {
        flat_search_gpu_gpu(base, N, query, std::min<size_t>(128,M), d, k, res);   // 暖机
        for (int r=0;r<R;++r) {
            CpuTimer t; t.start();
            if (strategy==0) flat_search_gpu_cpu(base, N, query, M, d, k, res);
            else             flat_search_gpu_gpu(base, N, query, M, d, k, res);
            gpu_ms += t.stop_ms();
        }
        gpu_ms /= R; recall = compute_recall(res, gt, gtd, k);

        CpuTimer tc; tc.start();
        flat_search_cpu(base, N, query, M, d, k, res_cpu);   // CPU 暴力基线
        cpu_ms = tc.stop_ms(); cpu_recall = compute_recall(res_cpu, gt, gtd, k);
    } else {
        std::printf("building IVF (nlist=%d)%s ...\n", nlist,
                    (need_global_pq||need_pc_pq)?" + PQ codebook":"");
        IVFIndex idx; build_ivf(idx, base, N, d, nlist);
        PQCodebook pq; pq.m=PQ_M; pq.ksub=PQ_KSUB; pq.subdim=(int)(d/PQ_M);
        if (need_global_pq) build_pq(pq, idx.reordered_base.data(), N, d, PQ_M, PQ_KSUB);
        if (need_pc_pq)     build_pq_percluster(pq, idx, idx.reordered_base.data(), N, d, PQ_M, PQ_KSUB);

        GpuCtx g; ivf_gpu_init(g, idx, pq, query, N, d, M, group_B, nprobe, std::max((int)k, rerank_k));
        std::vector<std::vector<int>> probe; coarse(g, idx, probe);
        const float* rbase = idx.reordered_base.data();
        auto groups = make_groups(group_mode, probe, query, M, d, group_B);

        run_groups(g, idx, pq, fine_mode, groups, probe, k, rerank_k, query, rbase, res);  // 暖机
        for (int r=0;r<R;++r) {
            CpuTimer t; t.start();
            overlap = run_groups(g, idx, pq, fine_mode, groups, probe, k, rerank_k, query, rbase, res);
            gpu_ms += t.stop_ms();
        }
        gpu_ms /= R; recall = compute_recall(res, gt, gtd, k);

        CpuTimer tc; tc.start();
        cpu_ivf_search(idx, query, M, k, nprobe, res_cpu);   // CPU IVF 基线
        cpu_ms = tc.stop_ms(); cpu_recall = compute_recall(res_cpu, gt, gtd, k);
        ivf_gpu_free(g);
    }

    // 输出
    std::printf("\nRecall    : GPU %.4f | CPU(ref) %.4f\n", recall, cpu_recall);
    std::printf("Latency   : GPU %.1f us/query (%.2f ms) | CPU %.2f ms | speedup %.2fx\n",
                gpu_ms/M*1000.0, gpu_ms, cpu_ms, cpu_ms/gpu_ms);
    if (!is_flat) {
        std::printf("params    : nlist=%d, nprobe=%d, group_B=%d, group=%s, overlap=%.3f",
                    nlist, nprobe, (int)group_B, gname[group_mode], overlap);
        if (need_global_pq||need_pc_pq)
            std::printf(", PQ_M=%d, PQ_KSUB=%d, rerank_k=%d", PQ_M, PQ_KSUB, rerank_k);
        std::printf("\n");
    }
    std::printf("====================\n");

    delete[] query; delete[] gt; delete[] base;
    return 0;
}
