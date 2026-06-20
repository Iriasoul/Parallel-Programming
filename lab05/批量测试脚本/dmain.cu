//  Lab5 GPU 编程  ANN 选题  —— 命令行参数版 (供 run_bench.py 批量调用)
/*
策略表:
  0 = Flat : GPU 矩阵乘(cuBLAS) + CPU 选 top-k          (GPU+CPU)
  1 = Flat : GPU 矩阵乘(cuBLAS) + GPU 选 top-k          (GPU+GPU)
  2 = IVF  (group 切换: 0 顺序 / 1 按主簇 / 2 按完整探查簇集合 / 3 query-KMeans)
  3 = IVF-PQ 全局PQ + 重排
  4 = IVF-PQ 簇内PQ + 重排

用法:
  ann                                  # 不带参数 = 用默认值跑一次 (人类可读 -> stderr)
  ann --strategy 2 --nlist 64 ...      # 命令行控制参数
  ann --header                         # 只打印 CSV 表头 (供脚本取列名)
说明: 信息打印走 stderr, 仅最后一行 CSV 数据走 stdout, 便于脚本采集。
编译: nvcc main.cu -o ann -O2 -lcublas -Xcompiler "/openmp /utf-8"
*/
#include "common.cuh"
#include "gpu_flat.cuh"
#include "gpu_ivf.cuh"
#include "gpu_ivf_pq.cuh"

static const char* CSV_HEADER =
    "strategy,name,nlist,nprobe,group_mode,group_name,group_B,rerank_k,"
    "PQ_M,PQ_KSUB,M,k,reps,recall,cpu_recall,gpu_ms,gpu_us_per_query,cpu_ms,speedup,overlap";

int main(int argc, char** argv) {
    // ---- 解析命令行 (无则用默认值) ----
    auto geti = [&](const char* key, int def)->int {
        for (int i=1;i+1<argc;++i) if (!std::strcmp(argv[i],key)) return std::atoi(argv[i+1]);
        return def;
    };
    auto hasflag = [&](const char* key)->bool {
        for (int i=1;i<argc;++i) if (!std::strcmp(argv[i],key)) return true;
        return false;
    };
    if (hasflag("--header")) { std::printf("%s\n", CSV_HEADER); return 0; }

    int    strategy    = geti("--strategy", 2);
    int    nlist       = geti("--nlist", 64);
    int    nprobe      = geti("--nprobe", 16);
    int    group_mode  = geti("--group_mode", 1);
    int    group_B     = geti("--group_B", 512);
    int    rerank_k    = geti("--rerank_k", 100);
    int    PQ_M        = geti("--pq_m", 8);
    int    PQ_KSUB     = geti("--pq_ksub", 256);
    int    test_number = geti("--testq", 2000);
    int    k_in        = geti("--k", 10);
    int    reps        = geti("--reps", 5);
    bool   do_cpu      = (geti("--cpu", 1) != 0);
    const size_t k     = (size_t)k_in;

    if (strategy<0 || strategy>4) { std::fprintf(stderr,"bad --strategy %d\n",strategy); return 1; }

    const std::string dir = ANN_DATA_DIR;
    size_t M_all=0, N=0, gtr=0, gtd=0, dq=0, db=0;
    float* query = LoadData<float>(dir + ANN_QUERY_FILE, M_all, dq);
    int*   gt    = LoadData<int>  (dir + ANN_GT_FILE,    gtr, gtd);
    float* base  = LoadData<float>(dir + ANN_BASE_FILE,  N, db);
    const size_t d = db;
    size_t M = std::min((size_t)test_number, std::min(M_all, gtr));
    if (d % PQ_M != 0) { std::fprintf(stderr,"ERROR: d=%d 不能被 PQ_M=%d 整除\n",(int)d,PQ_M); return 1; }

    bool is_flat        = (strategy<=1);
    bool need_global_pq = (strategy==3);
    bool need_pc_pq     = (strategy==4);
    int  fine_mode      = (strategy==2)?0 : (strategy==3)?1 : (strategy==4)?2 : -1;
    const char* sname[5] = {"Flat-GPU+CPU","Flat-GPU+GPU","IVF-exact",
                            "IVF-PQ-global+rerank","IVF-PQ-percluster+rerank"};
    const char* gname[4] = {"sequential","by-primary","by-probeset","query-kmeans"};

    { int dev=0; cudaDeviceProp p; CUDA_CHECK(cudaGetDevice(&dev));
      CUDA_CHECK(cudaGetDeviceProperties(&p,dev));
      std::fprintf(stderr,"GPU: %s | Strategy %d (%s)\n", p.name, strategy, sname[strategy]); }

    std::vector<std::vector<uint32_t>> res, res_cpu;
    std::vector<double> ts;
    double recall=0, gpu_ms=0, overlap=0, cpu_recall=-1, cpu_ms=-1;

    if (is_flat) {
        FlatCtx fc; flat_init(fc, base, N, query, M, d, k);              // 一次性准备, 不计时
        if (strategy==0) flat_search_gpu_cpu(fc, res); else flat_search_gpu_gpu(fc, res);  // 暖机
        for (int r=0;r<reps;++r) {
            CpuTimer t; t.start();
            if (strategy==0) flat_search_gpu_cpu(fc, res);
            else             flat_search_gpu_gpu(fc, res);
            ts.push_back(t.stop_ms());
        }
        recall = compute_recall(res, gt, gtd, k);
        if (do_cpu) { CpuTimer tc; tc.start();
            flat_search_cpu(base, N, query, M, d, k, res_cpu);
            cpu_ms = tc.stop_ms(); cpu_recall = compute_recall(res_cpu, gt, gtd, k); }
        flat_free(fc);
    } else {
        std::fprintf(stderr,"building IVF (nlist=%d)%s ...\n", nlist,
                     (need_global_pq||need_pc_pq)?" + PQ codebook":"");
        IVFIndex idx; build_ivf(idx, base, N, d, nlist);
        PQCodebook pq; pq.m=PQ_M; pq.ksub=PQ_KSUB; pq.subdim=(int)(d/PQ_M);
        if (need_global_pq) build_pq(pq, idx.reordered_base.data(), N, d, PQ_M, PQ_KSUB);
        if (need_pc_pq)     build_pq_percluster(pq, idx, idx.reordered_base.data(), N, d, PQ_M, PQ_KSUB);

        GpuCtx g; ivf_gpu_init(g, idx, pq, query, N, d, M, group_B, nprobe, std::max((int)k, rerank_k));
        const float* rbase = idx.reordered_base.data();
        std::vector<std::vector<int>> probe;

        // 暖机 (含粗排+分组+精排, 与计时口径一致)
        { coarse(g, idx, probe);
          auto gw = make_groups(group_mode, probe, query, M, d, group_B);
          run_groups(g, idx, pq, fine_mode, gw, probe, k, rerank_k, query, rbase, res); }
        for (int r=0;r<reps;++r) {
            CpuTimer t; t.start();
            coarse(g, idx, probe);                                               // 簇选(查询耗时)
            auto groups = make_groups(group_mode, probe, query, M, d, group_B);   // 分组(查询耗时)
            overlap = run_groups(g, idx, pq, fine_mode, groups, probe, k, rerank_k, query, rbase, res);
            ts.push_back(t.stop_ms());
        }
        recall = compute_recall(res, gt, gtd, k);
        if (do_cpu) { CpuTimer tc; tc.start();
            cpu_ivf_search(idx, query, M, k, nprobe, res_cpu);
            cpu_ms = tc.stop_ms(); cpu_recall = compute_recall(res_cpu, gt, gtd, k); }
        ivf_gpu_free(g);
    }

    std::sort(ts.begin(), ts.end());
    gpu_ms = ts[ts.size()/2];                 // 中位数
    double speedup = (cpu_ms>0) ? cpu_ms/gpu_ms : -1;

    std::fprintf(stderr,"recall=%.4f  gpu=%.2fms (median of %d)  cpu=%.2fms  speedup=%.2fx  overlap=%.3f\n",
                 recall, gpu_ms, reps, cpu_ms, speedup, overlap);

    // 仅这一行 -> stdout, 供脚本采集
    std::printf("%d,%s,%d,%d,%d,%s,%d,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.3f,%.2f,%.3f,%.3f,%.3f\n",
                strategy, sname[strategy], nlist, nprobe,
                is_flat?-1:group_mode, is_flat?"-":gname[group_mode], group_B, rerank_k,
                PQ_M, PQ_KSUB, (int)M, (int)k, reps,
                recall, cpu_recall, gpu_ms, gpu_ms/M*1000.0, cpu_ms, speedup, overlap);
    std::fflush(stdout);

    delete[] query; delete[] gt; delete[] base;
    return 0;
}
