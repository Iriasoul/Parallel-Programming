// ===========================================================================
//   0  = Flat-SIMD baseline        (Lab2 SIMD基线)
//   1  = Flat-SIMD + OpenMP        (base 分块并行)
//   2  = Flat-SIMD + Pthread       (手动线程)
//   3  = PQ-Rerank-SIMD baseline   (Lab2 逐 centroid + AoS, 单线程)
//   4  = PQ-Final baseline (单线程)(跨 centroid SIMD + SoA, 不含多线程)
//   5  = PQ-SIMD + OpenMP          (在 4 基础上加 OpenMP)
//   6  = PQ-SIMD + Pthread         (在 4 基础上加 Pthread)
//   7  = IVF-SIMD baseline         (单线程, IVF + 内存重排)
//   8  = IVF-SIMD + OpenMP         (簇级并行, dynamic schedule)
//   9  = IVF-SIMD + Pthread        (手动实现动态任务队列)
//   10 = IVF-PQ-SIMD baseline      (方法 1: 全局 PQ, 单线程)
//   11 = IVF-PQ-SIMD + OpenMP      (方法 1: 完整多线程流水线)
//   12 = IVF-PQ-SIMD + Pthread     (方法 1: 完整 pthread 流水线)
//   13 = IVF-PQ-v2-SIMD baseline   (方法 2: 簇内独立 PQ, 单线程)
//   14 = IVF-PQ-v2-SIMD + OpenMP   (方法 2: 簇级并行)
//   15 = IVF-PQ-v2-SIMD + Pthread  (方法 2: 簇级 pthread 动态调度)
//   16 = HNSW baseline             (hnswlib searchKnn, 单线程)
//   17 = HNSW 多入口 + OpenMP      (T 线程从不同起点贪心搜索, merge)
//   18 = HNSW 多入口 + Pthread     (同上, Pthread 版本)
// ===========================================================================

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <sys/time.h>
#include <omp.h>
#include <queue>
#include <cstdint>

#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"

// Lab3 各策略模块
#include "flat_thread.h"   // 策略 0, 1, 2
#include "pq_thread.h"     // 策略 3-6
#include "ivf.h"           // 策略 7-9
#include "ivf_pq.h"        // 策略 10-12 (方法 1: 全局 PQ)
#include "ivf_pq_v2.h"     // 策略 13-15 (方法 2: 簇内独立 PQ)
#include "hnsw.h"          // 策略 16-18 (进阶: HNSW)
#include "thread_pool.h"   // Pthread 线程池

using namespace hnswlib;

template<typename T>
T* LoadData(std::string data_path, size_t& n, size_t& d) {
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Failed to open " << data_path << std::endl;
        exit(-1);
    }
    fin.read((char*)&n, 4);
    fin.read((char*)&d, 4);
    T* data = new T[n * d];
    int sz = sizeof(T);
    for (size_t i = 0; i < n; ++i) {
        fin.read(((char*)data + i * d * sz), d * sz);
    }
    fin.close();
    std::cerr << "load data " << data_path << "\n";
    return data;
}

struct SearchResult {
    float recall;
    int64_t latency;
};

int main(int argc, char* argv[]) {
    // 离线索引统一固定为 4 线程
    const int OFFLINE_OMP_THREADS = 4;
    omp_set_num_threads(OFFLINE_OMP_THREADS);

    size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
    std::string data_path = "/anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt    = LoadData<int>  (data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base       = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    test_number = 2000;
    const size_t k = 10;
    std::vector<SearchResult> results(test_number);

    // 切换策略
    int strategy     = 16;    // 0..18
    int rerank_k     = 500;   // PQ rerank 用 (策略 3-6, 10-15)
    int nprobe       = 16;    // IVF 用 (策略 7-15)
    int nlist        = 256;   // IVF 簇数

    // 在线查询的并行配置 
    int online_omp_threads = 4;  
    int num_pthreads       = 4;  

    // 输出策略 + 在线查询使用的并行方式
    bool uses_omp     = (strategy==1||strategy==5||strategy==8||strategy==11
                       ||strategy==14||strategy==17);
    bool uses_pthread = (strategy==2||strategy==6||strategy==9||strategy==12
                       ||strategy==15||strategy==18);
    bool is_baseline  = (strategy==0||strategy==3||strategy==4||strategy==7
                       ||strategy==10||strategy==13||strategy==16);

    std::cout << "Running Strategy: " << strategy << "  ";
    if (uses_omp)          std::cout << "[Online: OpenMP, threads=" << online_omp_threads << "]";
    else if (uses_pthread) std::cout << "[Online: Pthread, threads=" << num_pthreads << "]";
    else if (is_baseline)  std::cout << "[Online: single-thread]";
    std::cout << "\n";

    // 离线预处理
    if (strategy >= 3 && strategy <= 6) {
        pq_offline(base, base_number, vecdim);
    }
    if (strategy >= 7 && strategy <= 9) {
        build_ivf(base, base_number, vecdim, nlist);
    }
    if (strategy >= 10 && strategy <= 12) {
        build_ivf(base, base_number, vecdim, nlist);
        build_ivf_pq(base_number, vecdim, 8);  // PQ_M = 8 (方法 1)
    }
    if (strategy >= 13 && strategy <= 15) {
        build_ivf(base, base_number, vecdim, nlist);
        build_ivf_pq_v2(base_number, vecdim, 8);  // PQ_M = 8 (方法 2)
    }
    if (strategy >= 16 && strategy <= 18) {
        // M=16, ef_construction=200, ef_search=64
        build_hnsw(base, base_number, vecdim, 16, 200, 64);
        if (strategy == 17 || strategy == 18) {
            // 子图并行需要子图索引
            int sub_count = (strategy == 17) ? online_omp_threads : num_pthreads;
            build_hnsw_sub(base, base_number, vecdim, sub_count, 16, 200, 64);
        }
    }

    // 在线搜索
    // 线程池在查询循环外一次性创建
    ThreadPool pthread_pool;
    if (uses_pthread) pthread_pool.init(num_pthreads);

    for (size_t i = 0; i < test_number; ++i) {
        struct timeval start, end;
        gettimeofday(&start, NULL);

        std::priority_queue<std::pair<float, uint32_t>> res;
        float* q = test_query + i * vecdim;

        switch (strategy) {
            case 0:  res = flat_search_simd       (base, q, base_number, vecdim, k); break;
            case 1:  res = flat_search_omp        (base, q, base_number, vecdim, k, online_omp_threads); break;
            case 2:  res = flat_search_pthread    (base, q, base_number, vecdim, k, num_pthreads, &pthread_pool); break;
            case 3:  res = pq_search_baseline     (q, base, base_number, vecdim, k, rerank_k); break;
            case 4:  res = pq_search_final_baseline(q, base, base_number, vecdim, k, rerank_k); break;
            case 5:  res = pq_search_thread       (q, base, base_number, vecdim, k, rerank_k, online_omp_threads); break;
            case 6:  res = pq_search_pthread      (q, base, base_number, vecdim, k, rerank_k, num_pthreads, &pthread_pool); break;
            case 7:  res = ivf_search_simd        (q, vecdim, k, nprobe); break;
            case 8:  res = ivf_search_omp         (q, vecdim, k, nprobe, online_omp_threads); break;
            case 9:  res = ivf_search_pthread     (q, vecdim, k, nprobe, num_pthreads, &pthread_pool); break;
            case 10: res = ivf_pq_search_simd     (q, vecdim, k, nprobe, rerank_k); break;
            case 11: res = ivf_pq_search_omp      (q, vecdim, k, nprobe, rerank_k, online_omp_threads); break;
            case 12: res = ivf_pq_search_pthread  (q, vecdim, k, nprobe, rerank_k, num_pthreads, &pthread_pool); break;
            case 13: res = ivf_pq2_search_simd    (q, vecdim, k, nprobe, rerank_k); break;
            case 14: res = ivf_pq2_search_omp     (q, vecdim, k, nprobe, rerank_k, online_omp_threads); break;
            case 15: res = ivf_pq2_search_pthread (q, vecdim, k, nprobe, rerank_k, num_pthreads, &pthread_pool); break;
            case 16: res = hnsw_search_baseline   (q, vecdim, k); break;
            case 17: res = hnsw_search_omp        (q, vecdim, k, online_omp_threads); break;
            case 18: res = hnsw_search_pthread    (q, vecdim, k, num_pthreads, &pthread_pool); break;
            default:
                std::cerr << "Unknown strategy: " << strategy << std::endl;
                return -1;
        }

        gettimeofday(&end, NULL);
        int64_t diff = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);

        std::set<uint32_t> gtset;
        for (size_t j = 0; j < k; ++j) gtset.insert(test_gt[j + i * test_gt_d]);

        size_t acc = 0;
        while (!res.empty()) {
            if (gtset.count(res.top().second)) ++acc;
            res.pop();
        }
        results[i] = { (float)acc / k, diff };
    }

    // 销毁线程池 
    if (uses_pthread) pthread_pool.destroy();

    float avg_recall = 0, avg_latency = 0;
    for (auto& r : results) { avg_recall += r.recall; avg_latency += r.latency; }
    std::cout << "Avg Recall: "  << avg_recall  / test_number
              << "\nAvg Latency: " << avg_latency / test_number << " us\n";

    // 打印参数
    if (strategy == 2 || strategy == 6 || strategy == 9 || strategy == 12 || strategy == 15 || strategy == 18)
        std::cout << "pthreads = " << num_pthreads << "\n";
    if (strategy >= 3 && strategy <= 6)  std::cout << "rerank_k = " << rerank_k << "\n";
    if (strategy >= 7 && strategy <= 9)  std::cout << "nlist = " << nlist << ", nprobe = " << nprobe << "\n";
    if (strategy >= 10 && strategy <= 15) {
        std::cout << "nlist = " << nlist << ", nprobe = " << nprobe
                  << ", rerank_k = " << rerank_k
                  << (strategy >= 13 ? " [IVF-PQ method 2]" : " [IVF-PQ method 1]") << "\n";
    }
    if (strategy >= 16 && strategy <= 18) {
        std::cout << "ef_search = " << hnsw_ns::ef_search << "\n";
    }
    std::cout << "\n====================\n";

    // 资源释放
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    if (strategy >= 3 && strategy <= 6)    pq_free();
    if (strategy >= 7 && strategy <= 15)   free_ivf();
    if (strategy >= 10 && strategy <= 12)  free_ivf_pq();
    if (strategy >= 13 && strategy <= 15)  free_ivf_pq_v2();
    if (strategy >= 16 && strategy <= 18)  free_hnsw();
    if (strategy == 17 || strategy == 18)  free_hnsw_sub();
    return 0;
}
