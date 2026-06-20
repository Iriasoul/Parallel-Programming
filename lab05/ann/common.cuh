#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <queue>
#include <set>
#include <chrono>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <utility>

#include <cuda_runtime.h>
#include <cublas_v2.h>

// 数据集目录
#ifndef ANN_DATA_DIR
#define ANN_DATA_DIR "D:/lab05-GPU/ann/data/"
#endif

// 数据文件名
#define ANN_BASE_FILE  "DEEP100K.base.100k.fbin"
#define ANN_QUERY_FILE "DEEP100K.query.fbin"
#define ANN_GT_FILE    "DEEP100K.gt.query.100k.top100.bin"

// 错误检查
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "[CUDA error] %s:%d : %s\n",                 \
                         __FILE__, __LINE__, cudaGetErrorString(_e));         \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t _s = (call);                                           \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                    \
            std::fprintf(stderr, "[cuBLAS error] %s:%d : status %d\n",        \
                         __FILE__, __LINE__, (int)_s);                        \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)


// 读取
template <typename T>
T* LoadData(const std::string& path, size_t& n, size_t& d) {
    std::ifstream fin(path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Failed to open " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    uint32_t n32 = 0, d32 = 0;
    fin.read(reinterpret_cast<char*>(&n32), 4);
    fin.read(reinterpret_cast<char*>(&d32), 4);
    n = n32; d = d32;
    T* data = new T[n * d];
    fin.read(reinterpret_cast<char*>(data),
             static_cast<std::streamsize>(n * d * sizeof(T)));
    fin.close();
    std::cerr << "load " << path << "  n=" << n << " d=" << d << "\n";
    return data;
}

// CPU 计时器
struct CpuTimer {
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double stop_ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// GPU 计时器 (基于 cudaEvent)
struct GpuTimer {
    cudaEvent_t a, b;
    GpuTimer()  { cudaEventCreate(&a); cudaEventCreate(&b); }
    ~GpuTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start() { cudaEventRecord(a, 0); }
    float stop_ms() {
        cudaEventRecord(b, 0);
        cudaEventSynchronize(b);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        return ms;
    }
};

// 召回率
inline double compute_recall(const std::vector<std::vector<uint32_t>>& results,
                             const int* gt, size_t gt_d, size_t k) {
    if (results.empty()) return 0.0;
    double total = 0.0;
    const size_t M = results.size();
    for (size_t i = 0; i < M; ++i) {
        std::set<int> gtset;
        for (size_t j = 0; j < k; ++j) gtset.insert(gt[i * gt_d + j]);
        size_t hit = 0;
        for (uint32_t id : results[i])
            if (gtset.count(static_cast<int>(id))) ++hit;
        total += static_cast<double>(hit) / static_cast<double>(k);
    }
    return total / static_cast<double>(M);
}


// CPU 端取下标
inline void topk_from_dist(const float* dist, size_t N, size_t k,
                           std::vector<uint32_t>& out) {
    std::priority_queue<std::pair<float, uint32_t>> heap; // 大顶堆 (按距离)
    for (size_t j = 0; j < N; ++j) {
        float dd = dist[j];
        if (heap.size() < k)        heap.push({dd, (uint32_t)j});
        else if (dd < heap.top().first) { heap.pop(); heap.push({dd, (uint32_t)j}); }
    }
    out.resize(heap.size());
    size_t idx = heap.size();
    while (!heap.empty()) { out[--idx] = heap.top().second; heap.pop(); }
}
