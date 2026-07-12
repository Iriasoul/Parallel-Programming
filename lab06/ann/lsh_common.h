// lsh_common.h 
// 局部敏感哈希(SimHash / 符号随机投影)

#pragma once
#include "bench_common.h"
#include <cmath>

// 生成 K 个 d 维高斯随机超平面（行主序 planes[K*d]），固定种子可复现
inline void lsh_gen_planes(int K, int d, std::vector<float>& planes, unsigned seed = 12345u) {
    planes.resize((size_t)K * d);
    srand(seed);
    auto uni = [&]() { return (rand() + 1.0) / ((double)RAND_MAX + 2.0); };  // (0,1)
    const double TWO_PI = 6.283185307179586;
    for (size_t i = 0; i < planes.size(); i += 2) {                          // Box-Muller
        double u1 = uni(), u2 = uni();
        double r = std::sqrt(-2.0 * std::log(u1)), t = TWO_PI * u2;
        planes[i] = (float)(r * std::cos(t));
        if (i + 1 < planes.size()) planes[i + 1] = (float)(r * std::sin(t));
    }
}

// 64 位 popcount（SSE4.2）
static inline int lsh_popc64(uint64_t x) {
#if BENCH_HAVE_AVX2
    return (int)_mm_popcnt_u64(x);
#else
    int c = 0; while (x) { x &= x - 1; ++c; } return c;
#endif
}

// 把 K 个投影值的符号位打包进 nwords 个 uint64（proj>=0 则置 1）
static inline void lsh_pack_signs(const float* proj, int K, uint64_t* code) {
    int nwords = K / 64;
    for (int w = 0; w < nwords; ++w) {
        uint64_t word = 0;
        for (int b = 0; b < 64; ++b)
            if (proj[w * 64 + b] >= 0.0f) word |= (1ull << b);
        code[w] = word;
    }
}

// 两个 K-bit 码的 Hamming 距离
static inline int lsh_hamming(const uint64_t* a, const uint64_t* b, int nwords) {
    int h = 0;
    for (int w = 0; w < nwords; ++w) h += lsh_popc64(a[w] ^ b[w]);
    return h;
}
