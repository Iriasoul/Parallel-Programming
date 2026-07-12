// 测试底座。说明：SIMD 此处重写为 AVX2
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

#if defined(__AVX2__)
  #include <immintrin.h>
  #define BENCH_HAVE_AVX2 1
#else
  #define BENCH_HAVE_AVX2 0
#endif

// 数据文件名
#define ANN_BASE_FILE  "DEEP100K.base.100k.fbin"
#define ANN_QUERY_FILE "DEEP100K.query.fbin"
#define ANN_GT_FILE    "DEEP100K.gt.query.100k.top100.bin"

// 暴力算法基线（avx2 flat, 见 strategy 0）
static const double BENCH_BASELINE_US = 1562.5;

// 读取 fbin
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

// 内积内核
// 标量（对照 / SIMD 关闭时用）
static inline float ip_scalar(const float* a, const float* b, size_t dim) {
    float s = 0.0f;
    for (size_t i = 0; i < dim; ++i) s += a[i] * b[i];
    return s;
}

#if BENCH_HAVE_AVX2
// AVX2 + FMA：一次 8 路
static inline float ip_avx2(const float* a, const float* b, size_t dim) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float r = _mm_cvtss_f32(s);
    for (; i < dim; ++i) r += a[i] * b[i];
    return r;
}
static inline float ip_default(const float* a, const float* b, size_t dim) { return ip_avx2(a, b, dim); }
#else
static inline float ip_default(const float* a, const float* b, size_t dim) { return ip_scalar(a, b, dim); }
#endif

// L2 平方距离（KMeans），AVX2
#if BENCH_HAVE_AVX2
static inline float l2sq_avx2(const float* a, const float* b, size_t dim) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 df = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc = _mm256_fmadd_ps(df, df, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float r = _mm_cvtss_f32(s);
    for (; i < dim; ++i) { float t = a[i] - b[i]; r += t * t; }
    return r;
}
static inline float l2sq_default(const float* a, const float* b, size_t dim) { return l2sq_avx2(a, b, dim); }
#else
static inline float l2sq_default(const float* a, const float* b, size_t dim) {
    float r = 0.0f;
    for (size_t i = 0; i < dim; ++i) { float t = a[i] - b[i]; r += t * t; }
    return r;
}
#endif

// 计时
struct CpuTimer {
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double stop_ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double stop_us() const { return stop_ms() * 1000.0; }
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

// Top-k 
using MaxHeap = std::priority_queue<std::pair<float, uint32_t>>;

inline void heap_push(MaxHeap& h, float dis, uint32_t id, size_t k) {
    if (h.size() < k)               h.push({dis, id});
    else if (dis < h.top().first) { h.pop(); h.push({dis, id}); }
}

// 堆 -> 升序 id 列表
inline std::vector<uint32_t> heap_to_ids(MaxHeap& h) {
    std::vector<uint32_t> out(h.size());
    size_t idx = h.size();
    while (!h.empty()) { out[--idx] = h.top().second; h.pop(); }
    return out;
}

// 堆 -> (dist,id) 列表, 用于线程内局部结果归并
inline std::vector<std::pair<float, uint32_t>> heap_to_ids_pairs(MaxHeap& h) {
    std::vector<std::pair<float, uint32_t>> out;
    out.reserve(h.size());
    while (!h.empty()) { out.push_back(h.top()); h.pop(); }
    return out;
}
