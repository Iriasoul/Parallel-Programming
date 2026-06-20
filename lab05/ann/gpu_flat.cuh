#pragma once

#include "gpu_common.cuh"

inline void flat_search_cpu(const float* base, size_t N, const float* query, size_t M,
                            size_t d, size_t k, std::vector<std::vector<uint32_t>>& results) {
    results.assign(M, {});
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)M; ++i) {
        const float* q = query + (size_t)i*d;
        std::priority_queue<std::pair<float,uint32_t>> heap;
        for (size_t j = 0; j < N; ++j) {
            const float* b = base + j*d;
            float ip = 0.0f; for (size_t l = 0; l < d; ++l) ip += q[l]*b[l];
            float dist = 1.0f - ip;
            if (heap.size() < k) heap.push({dist,(uint32_t)j});
            else if (dist < heap.top().first){ heap.pop(); heap.push({dist,(uint32_t)j}); }
        }
        std::vector<uint32_t> out(heap.size()); size_t p = heap.size();
        while(!heap.empty()){ out[--p]=heap.top().second; heap.pop(); }
        results[i] = std::move(out);
    }
}

// 持久化上下文: base/query/显存/句柄一次性准备好 
struct FlatCtx {
    cublasHandle_t handle=nullptr;
    float* dB=nullptr; float* dQ=nullptr; float* dS=nullptr;
    uint32_t* dIdx=nullptr; float* dDist=nullptr;
    size_t N=0,d=0,M=0,k=0;
};
inline void flat_init(FlatCtx& c, const float* base, size_t N,
                      const float* query, size_t M, size_t d, size_t k) {
    c.N=N;c.d=d;c.M=M;c.k=k;
    CUBLAS_CHECK(cublasCreate(&c.handle));
    CUDA_CHECK(cudaMalloc(&c.dB, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.dQ, M*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.dS, M*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.dIdx, M*k*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&c.dDist, M*k*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(c.dB, base,  N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c.dQ, query, M*d*sizeof(float), cudaMemcpyHostToDevice));
}
inline void flat_free(FlatCtx& c) {
    cudaFree(c.dB);cudaFree(c.dQ);cudaFree(c.dS);cudaFree(c.dIdx);cudaFree(c.dDist);
    cublasDestroy(c.handle);
}
// dS = Q×B^T 并转距离
inline void flat_gemm(FlatCtx& c) {
    const float alpha=1.f, beta=0.f;
    CUBLAS_CHECK(cublasSgemm(c.handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)c.N,(int)c.M,(int)c.d,
                             &alpha, c.dB,(int)c.d, c.dQ,(int)c.d, &beta, c.dS,(int)c.N));
    int nt=256; size_t total=c.M*c.N;
    ip_to_dist_kernel<<<(int)((total+nt-1)/nt),nt>>>(c.dS, total);
}
inline void flat_search_gpu_cpu(FlatCtx& c, std::vector<std::vector<uint32_t>>& results) {
    flat_gemm(c);
    std::vector<float> hS(c.M*c.N);
    CUDA_CHECK(cudaMemcpy(hS.data(), c.dS, c.M*c.N*sizeof(float), cudaMemcpyDeviceToHost));
    results.assign(c.M, {});
    #pragma omp parallel for schedule(dynamic)
    for (int i=0;i<(int)c.M;++i) topk_from_dist(hS.data()+(size_t)i*c.N, c.N, c.k, results[i]);
}
inline void flat_search_gpu_gpu(FlatCtx& c, std::vector<std::vector<uint32_t>>& results) {
    flat_gemm(c);
    int nt=256; size_t sh=(size_t)nt*(sizeof(float)+sizeof(int));
    topk_kernel<<<(int)c.M, nt, sh>>>(c.dS, (int)c.N, (int)c.k, c.dIdx, c.dDist);
    CUDA_CHECK(cudaGetLastError());
    std::vector<uint32_t> hIdx(c.M*c.k);
    CUDA_CHECK(cudaMemcpy(hIdx.data(), c.dIdx, c.M*c.k*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    results.assign(c.M, {});
    for (size_t i=0;i<c.M;++i) results[i].assign(hIdx.begin()+i*c.k, hIdx.begin()+(i+1)*c.k);
}
