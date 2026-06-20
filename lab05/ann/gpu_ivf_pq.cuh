#pragma once

#include "gpu_ivf.cuh"

// 全局 PQ
inline void build_pq(PQCodebook& pq, const float* rbase, size_t N, size_t d, int m, int ksub) {
    pq.m=m; pq.ksub=ksub; pq.subdim=(int)(d/m); int sd=pq.subdim;
    pq.codebook.assign((size_t)m*ksub*sd, 0.f);
    pq.codes.assign(N*m, 0);
    size_t train_n = std::min<size_t>(20000, N);
    size_t stride = std::max<size_t>(1, N/train_n);
    std::vector<float> sub(train_n*sd);
    for (int j=0;j<m;++j) {
        for (size_t i=0;i<train_n;++i)
            std::memcpy(&sub[i*sd], rbase + (i*stride)*d + (size_t)j*sd, sd*sizeof(float));
        kmeans(sub.data(), train_n, sd, ksub, pq.codebook.data()+(size_t)j*ksub*sd, 12, (unsigned)(100+j));
        const float* cb = pq.codebook.data()+(size_t)j*ksub*sd;
        #pragma omp parallel for schedule(static)
        for (long long i=0;i<(long long)N;++i) {
            const float* x = rbase + i*d + (size_t)j*sd;
            float bd=1e30f; int bc=0;
            for (int t=0;t<ksub;++t){ float dd=l2_cpu(x, cb+(size_t)t*sd, sd); if(dd<bd){bd=dd;bc=t;} }
            pq.codes[(size_t)i*m + j] = (uint8_t)bc;
        }
    }
}

// 簇内 PQ
inline void build_pq_percluster(PQCodebook& pq, const IVFIndex& idx, const float* rbase,
                                size_t N, size_t d, int m, int ksub) {
    pq.m=m; pq.ksub=ksub; pq.subdim=(int)(d/m); int sd=pq.subdim;
    pq.codebook_pc.assign((size_t)idx.nlist*m*ksub*sd, 0.f);
    pq.codes_pc.assign(N*m, 0);
    #pragma omp parallel for schedule(dynamic,1)
    for (int c=0;c<idx.nlist;++c) {
        size_t s=idx.cluster_starts[c], e=idx.cluster_starts[c+1], csize=e-s;
        if (csize==0) continue;
        float* cb_c = pq.codebook_pc.data() + (size_t)c*m*ksub*sd;
        std::vector<float> sub(csize*sd);
        for (int j=0;j<m;++j) {
            for (size_t i=0;i<csize;++i)
                std::memcpy(&sub[i*sd], rbase + (s+i)*d + (size_t)j*sd, sd*sizeof(float));
            kmeans(sub.data(), csize, sd, ksub, cb_c + (size_t)j*ksub*sd, 15, (unsigned)(c*131+j));
            const float* cbj = cb_c + (size_t)j*ksub*sd;
            for (size_t i=0;i<csize;++i) {
                const float* x = rbase + (s+i)*d + (size_t)j*sd;
                float bd=1e30f; int bc=0;
                for (int t=0;t<ksub;++t){ float dd=l2_cpu(x, cbj+(size_t)t*sd, sd); if(dd<bd){bd=dd;bc=t;} }
                pq.codes_pc[(s+i)*m + j] = (uint8_t)bc;
            }
        }
    }
}
