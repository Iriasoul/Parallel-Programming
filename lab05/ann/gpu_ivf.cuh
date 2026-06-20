#pragma once

#include "gpu_common.cuh"
#include <unordered_set>
#include <unordered_map>
#include <random>

// IVF 结构体
struct IVFIndex {
    int nlist = 0; size_t d = 0;
    std::vector<float>    centroids, reordered_base;
    std::vector<uint32_t> reordered_ids;
    std::vector<size_t>   cluster_starts;
};
struct PQCodebook {
    int m = 8, ksub = 256, subdim = 0;
    std::vector<float>   codebook;     // 全局: m*ksub*subdim
    std::vector<uint8_t> codes;        // 全局编码: N*m
    std::vector<float>   codebook_pc;  // 簇内: nlist*m*ksub*subdim
    std::vector<uint8_t> codes_pc;     // 簇内编码: N*m
};

static inline float l2_cpu(const float* a, const float* b, size_t d) {
    float s=0; for (size_t i=0;i<d;++i){ float t=a[i]-b[i]; s+=t*t; } return s;
}
static inline float ip_cpu(const float* a, const float* b, size_t d) {
    float s=0; for (size_t i=0;i<d;++i) s+=a[i]*b[i]; return s;
}
inline void kmeans(const float* data, size_t n, size_t d, int nlist, float* cents,
                   int iters, unsigned seed=123) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> pick(0, n-1);
    for (int i=0;i<nlist;++i) std::memcpy(cents+i*d, data+pick(rng)*d, d*sizeof(float));
    std::vector<int> assign(n,0);
    for (int it=0; it<iters; ++it) {
        #pragma omp parallel for schedule(static)
        for (long long i=0;i<(long long)n;++i) {
            float bd=1e30f; int bc=0;
            for (int c=0;c<nlist;++c){ float d2=l2_cpu(data+i*d,cents+c*d,d); if(d2<bd){bd=d2;bc=c;} }
            assign[i]=bc;
        }
        std::vector<float> next(nlist*d,0.f); std::vector<int> cnt(nlist,0);
        for (size_t i=0;i<n;++i){ int c=assign[i]; cnt[c]++; for(size_t j=0;j<d;++j) next[c*d+j]+=data[i*d+j]; }
        for (int c=0;c<nlist;++c) if(cnt[c]) for(size_t j=0;j<d;++j) cents[c*d+j]=next[c*d+j]/cnt[c];
    }
}
inline void build_ivf(IVFIndex& idx, const float* base, size_t N, size_t d, int nlist) {
    idx.d=d; idx.nlist=nlist; idx.centroids.assign((size_t)nlist*d,0.f);
    kmeans(base, std::min<size_t>(20000,N), d, nlist, idx.centroids.data(), 15);
    std::vector<int> assign(N);
    #pragma omp parallel for schedule(static)
    for (long long i=0;i<(long long)N;++i){
        float bd=1e30f; int bc=0;
        for (int c=0;c<nlist;++c){ float d2=l2_cpu(base+i*d, idx.centroids.data()+c*d, d); if(d2<bd){bd=d2;bc=c;} }
        assign[i]=bc;
    }
    idx.cluster_starts.assign(nlist+1,0);
    for (size_t i=0;i<N;++i) idx.cluster_starts[assign[i]+1]++;
    for (int c=1;c<=nlist;++c) idx.cluster_starts[c]+=idx.cluster_starts[c-1];
    idx.reordered_base.assign(N*d,0.f); idx.reordered_ids.assign(N,0);
    std::vector<size_t> ptr(idx.cluster_starts.begin(), idx.cluster_starts.begin()+nlist);
    for (size_t i=0;i<N;++i){ int c=assign[i]; size_t pos=ptr[c]++;
        std::memcpy(idx.reordered_base.data()+pos*d, base+i*d, d*sizeof(float));
        idx.reordered_ids[pos]=(uint32_t)i; }
}
inline void cpu_ivf_search(const IVFIndex& idx, const float* query, size_t M, size_t k, int nprobe,
                           std::vector<std::vector<uint32_t>>& results) {
    const size_t d=idx.d; results.assign(M,{});
    #pragma omp parallel for schedule(dynamic)
    for (int qi=0; qi<(int)M; ++qi) {
        const float* q=query+(size_t)qi*d;
        std::priority_queue<std::pair<float,int>> coarse;
        for (int c=0;c<idx.nlist;++c){ float dis=1.f-ip_cpu(q, idx.centroids.data()+(size_t)c*d, d);
            if((int)coarse.size()<nprobe||dis<coarse.top().first){ coarse.push({dis,c}); if((int)coarse.size()>nprobe) coarse.pop(); } }
        std::priority_queue<std::pair<float,uint32_t>> heap;
        while(!coarse.empty()){ int c=coarse.top().second; coarse.pop();
            size_t s=idx.cluster_starts[c], e=idx.cluster_starts[c+1];
            for(size_t i=s;i<e;++i){ float dis=1.f-ip_cpu(q, idx.reordered_base.data()+i*d, d);
                if(heap.size()<k||dis<heap.top().first){ heap.push({dis, idx.reordered_ids[i]}); if(heap.size()>k) heap.pop(); } } }
        std::vector<uint32_t> out(heap.size()); size_t p=heap.size();
        while(!heap.empty()){ out[--p]=heap.top().second; heap.pop(); }
        results[qi]=std::move(out);
    }
}

// GPU 上下文
struct GpuCtx {
    cublasHandle_t handle=nullptr;
    float* dRB=nullptr; float* dC=nullptr; float* dQ=nullptr;
    float* dQg=nullptr; int* dGidx=nullptr;
    float* dScC=nullptr; uint32_t* dCoIdx=nullptr; float* dCoDist=nullptr;
    float* dSc=nullptr; uint32_t* dCaIdx=nullptr; float* dCaDist=nullptr;
    float* dCodebook=nullptr; uint8_t* dCodes=nullptr; float* dLUT=nullptr;     // 全局 PQ
    float* dCodebookPC=nullptr; uint8_t* dCodesPC=nullptr;                       // 簇内 PQ
    size_t N=0,d=0,M=0,maxCl=0,maxG=0; int nprobe=0, maxCand=0;
};
inline void ivf_gpu_init(GpuCtx& g, const IVFIndex& idx, const PQCodebook& pq, const float* query,
                         size_t N, size_t d, size_t M, size_t group_B, int nprobe, int maxCand) {
    g.N=N;g.d=d;g.M=M;g.maxG=group_B;g.nprobe=nprobe;g.maxCand=maxCand;g.maxCl=0;
    for(int c=0;c<idx.nlist;++c) g.maxCl=std::max(g.maxCl, idx.cluster_starts[c+1]-idx.cluster_starts[c]);
    size_t cb = (size_t)pq.m*pq.ksub*pq.subdim;
    CUBLAS_CHECK(cublasCreate(&g.handle));
    CUDA_CHECK(cudaMalloc(&g.dRB, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.dC, (size_t)idx.nlist*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.dQ, M*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.dQg, group_B*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.dGidx, group_B*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.dScC, M*(size_t)idx.nlist*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.dCoIdx, M*(size_t)nprobe*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&g.dCoDist, M*(size_t)nprobe*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.dSc, group_B*g.maxCl*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.dCaIdx, group_B*(size_t)maxCand*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&g.dCaDist, group_B*(size_t)maxCand*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.dLUT, group_B*(size_t)pq.m*pq.ksub*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.dCodebook, cb*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.dCodes, N*(size_t)pq.m*sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&g.dCodebookPC, (size_t)idx.nlist*cb*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.dCodesPC, N*(size_t)pq.m*sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(g.dRB, idx.reordered_base.data(), N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.dC, idx.centroids.data(), (size_t)idx.nlist*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.dQ, query, M*d*sizeof(float), cudaMemcpyHostToDevice));
    // 仅上传已训练好的码本 (未用到的策略其码本为空, 跳过)
    if (!pq.codebook.empty())
        CUDA_CHECK(cudaMemcpy(g.dCodebook, pq.codebook.data(), cb*sizeof(float), cudaMemcpyHostToDevice));
    if (!pq.codes.empty())
        CUDA_CHECK(cudaMemcpy(g.dCodes, pq.codes.data(), N*(size_t)pq.m*sizeof(uint8_t), cudaMemcpyHostToDevice));
    if (!pq.codebook_pc.empty())
        CUDA_CHECK(cudaMemcpy(g.dCodebookPC, pq.codebook_pc.data(), (size_t)idx.nlist*cb*sizeof(float), cudaMemcpyHostToDevice));
    if (!pq.codes_pc.empty())
        CUDA_CHECK(cudaMemcpy(g.dCodesPC, pq.codes_pc.data(), N*(size_t)pq.m*sizeof(uint8_t), cudaMemcpyHostToDevice));
}
inline void ivf_gpu_free(GpuCtx& g) {
    cudaFree(g.dRB);cudaFree(g.dC);cudaFree(g.dQ);cudaFree(g.dQg);cudaFree(g.dGidx);
    cudaFree(g.dScC);cudaFree(g.dCoIdx);cudaFree(g.dCoDist);
    cudaFree(g.dSc);cudaFree(g.dCaIdx);cudaFree(g.dCaDist);cudaFree(g.dLUT);
    cudaFree(g.dCodebook);cudaFree(g.dCodes);cudaFree(g.dCodebookPC);cudaFree(g.dCodesPC);
    cublasDestroy(g.handle);
}

// 粗排：每个 query 取 nprobe 个最近簇 - probe[M][nprobe]
inline void coarse(GpuCtx& g, const IVFIndex& idx, std::vector<std::vector<int>>& probe) {
    const float alpha=1.f,beta=0.f; int nt=256, nprobe=g.nprobe;
    CUBLAS_CHECK(cublasSgemm(g.handle, CUBLAS_OP_T, CUBLAS_OP_N, idx.nlist,(int)g.M,(int)g.d,
                             &alpha, g.dC,(int)g.d, g.dQ,(int)g.d, &beta, g.dScC, idx.nlist));
    ip_to_dist_kernel<<<(int)((g.M*idx.nlist+nt-1)/nt),nt>>>(g.dScC, g.M*idx.nlist);
    size_t sh=(size_t)nt*(sizeof(float)+sizeof(int));
    topk_kernel<<<(int)g.M,nt,sh>>>(g.dScC, idx.nlist, nprobe, g.dCoIdx, g.dCoDist);
    CUDA_CHECK(cudaGetLastError());
    std::vector<uint32_t> hco(g.M*nprobe);
    CUDA_CHECK(cudaMemcpy(hco.data(), g.dCoIdx, g.M*(size_t)nprobe*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    probe.assign(g.M,{});
    for(size_t i=0;i<g.M;++i){ probe[i].resize(nprobe); for(int j=0;j<nprobe;++j) probe[i][j]=(int)hco[i*nprobe+j]; }
}

// 精排：mode 0 精确 / 1 全局PQ+重排 / 2 簇内PQ+重排 
inline void fine_group(GpuCtx& g, const IVFIndex& idx, const PQCodebook& pq, int mode,
                       const std::vector<int>& group, const std::vector<std::vector<int>>& probe,
                       std::vector<std::vector<uint32_t>>& results,
                       size_t k, int rerank_k, const float* query_host, const float* rbase_host,
                       double& sum_nc, double& tiles) {
    int B=(int)group.size(); if(B<=0){ sum_nc=tiles=0; return; } int nt=256;
    bool use_adc=(mode>=1), per_cluster=(mode==2);   // 1=全局PQ, 2=簇内PQ; 均带重排
    int kc_cap = use_adc ? rerank_k : (int)k;        // PQ 模式多留候选用于重排
    const uint8_t* codes_base = per_cluster ? g.dCodesPC : g.dCodes;
    size_t cb_stride=(size_t)pq.m*pq.ksub*pq.subdim;

    std::vector<int> gidx(group.begin(), group.end());
    CUDA_CHECK(cudaMemcpy(g.dGidx, gidx.data(), B*sizeof(int), cudaMemcpyHostToDevice));
    gather_rows<<<B,nt>>>(g.dQ, g.dQg, g.dGidx, B, (int)g.d);

    if (use_adc && !per_cluster) {
        long tot=(long)B*pq.m*pq.ksub;
        build_lut<<<(int)((tot+nt-1)/nt),nt>>>(g.dQg, g.dCodebook, g.dLUT, B, pq.m, pq.ksub, pq.subdim);
        CUDA_CHECK(cudaGetLastError());
    }

    std::unordered_map<int,std::vector<int>> members;
    for(int li=0;li<B;++li) for(int c:probe[group[li]]) members[c].push_back(li);
    sum_nc=(double)B*g.nprobe; tiles=(double)B*(double)members.size();

    const float alpha=1.f,beta=0.f;
    std::vector<uint32_t> hCaIdx((size_t)B*kc_cap); std::vector<float> hCaDist((size_t)B*kc_cap);
    std::vector<std::priority_queue<std::pair<float,uint32_t>>> heaps(B);

    for (auto& kv: members) {
        int c=kv.first; const std::vector<int>& mem=kv.second;
        size_t s=idx.cluster_starts[c], e=idx.cluster_starts[c+1];
        int vc=(int)(e-s); if(vc<=0) continue; int kc=std::min(kc_cap,vc);
        if (!use_adc) {
            CUBLAS_CHECK(cublasSgemm(g.handle, CUBLAS_OP_T, CUBLAS_OP_N, vc, B, (int)g.d,
                                     &alpha, g.dRB+s*g.d,(int)g.d, g.dQg,(int)g.d, &beta, g.dSc, vc));
            ip_to_dist_kernel<<<(int)(((size_t)B*vc+nt-1)/nt),nt>>>(g.dSc,(size_t)B*vc);
        } else {
            if (per_cluster) {
                long tot=(long)B*pq.m*pq.ksub;
                build_lut<<<(int)((tot+nt-1)/nt),nt>>>(g.dQg, g.dCodebookPC+(size_t)c*cb_stride,
                                                       g.dLUT, B, pq.m, pq.ksub, pq.subdim);
            }
            adc_kernel<<<(int)(((long)B*vc+nt-1)/nt),nt>>>(codes_base+s*pq.m, g.dLUT, g.dSc, B, vc, pq.m, pq.ksub);
        }
        size_t sh=(size_t)nt*(sizeof(float)+sizeof(int));
        topk_kernel<<<B,nt,sh>>>(g.dSc, vc, kc, g.dCaIdx, g.dCaDist);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(hCaIdx.data(), g.dCaIdx, (size_t)B*kc*sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hCaDist.data(), g.dCaDist, (size_t)B*kc*sizeof(float), cudaMemcpyDeviceToHost));
        for (int li: mem){ auto& h=heaps[li];
            for(int j=0;j<kc;++j){ float dis=hCaDist[(size_t)li*kc+j]; if(dis==FLT_MAX) continue;
                size_t pos = s + hCaIdx[(size_t)li*kc+j];
                uint32_t val = use_adc ? (uint32_t)pos : idx.reordered_ids[pos];
                if((int)h.size()<kc_cap||dis<h.top().first){ h.push({dis,val}); if((int)h.size()>kc_cap) h.pop(); } } }
    }

    for (int li=0; li<B; ++li) {
        auto& h = heaps[li];
        if (!use_adc) {
            std::vector<uint32_t> out(h.size()); size_t p=h.size();
            while(!h.empty()){ out[--p]=h.top().second; h.pop(); }
            results[group[li]] = std::move(out);
        } else {
            const float* q = query_host + (size_t)group[li]*g.d;   // PQ 候选用全精度向量精确重排
            std::priority_queue<std::pair<float,uint32_t>> fin;
            while(!h.empty()){
                size_t pos = h.top().second; h.pop();
                float dis = 1.f - ip_cpu(q, rbase_host + pos*g.d, g.d);
                uint32_t id = idx.reordered_ids[pos];
                if(fin.size()<k||dis<fin.top().first){ fin.push({dis,id}); if(fin.size()>k) fin.pop(); }
            }
            std::vector<uint32_t> out(fin.size()); size_t p=fin.size();
            while(!fin.empty()){ out[--p]=fin.top().second; fin.pop(); }
            results[group[li]] = std::move(out);
        }
    }
}

// 四种分组策略
inline std::vector<std::vector<int>> group_sequential(size_t M, size_t B) {
    std::vector<std::vector<int>> gs;
    for (size_t s=0;s<M;s+=B){ std::vector<int> g; for(size_t i=s;i<std::min(s+B,M);++i) g.push_back((int)i); gs.push_back(std::move(g)); }
    return gs;
}
inline std::vector<std::vector<int>> group_by_primary(const std::vector<std::vector<int>>& probe, size_t M, size_t B) {
    std::vector<int> order(M); for(size_t i=0;i<M;++i) order[i]=(int)i;
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){ return probe[a][0]<probe[b][0]; });
    std::vector<std::vector<int>> gs;
    for (size_t s=0;s<M;s+=B){ std::vector<int> g; for(size_t i=s;i<std::min(s+B,M);++i) g.push_back(order[i]); gs.push_back(std::move(g)); }
    return gs;
}
inline std::vector<std::vector<int>> group_by_probeset(const std::vector<std::vector<int>>& probe, size_t M, size_t B) {
    std::vector<std::vector<int>> sig(M);
    for(size_t i=0;i<M;++i){ sig[i]=probe[i]; std::sort(sig[i].begin(),sig[i].end()); }
    std::vector<int> order(M); for(size_t i=0;i<M;++i) order[i]=(int)i;
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){ return sig[a]<sig[b]; });
    std::vector<std::vector<int>> gs;
    for (size_t s=0;s<M;s+=B){ std::vector<int> g; for(size_t i=s;i<std::min(s+B,M);++i) g.push_back(order[i]); gs.push_back(std::move(g)); }
    return gs;
}
inline std::vector<std::vector<int>> group_query_kmeans(const float* query, size_t M, size_t d, size_t B) {
    int G=(int)((M+B-1)/B); std::vector<float> cents((size_t)G*d);
    std::mt19937 rng(7); std::uniform_int_distribution<size_t> pick(0,M-1);
    for(int c=0;c<G;++c) std::memcpy(&cents[c*d], query+pick(rng)*d, d*sizeof(float));
    std::vector<int> asn(M,0);
    for(int it=0;it<5;++it){
        #pragma omp parallel for schedule(static)
        for(long long i=0;i<(long long)M;++i){ float bd=1e30f; int bc=0;
            for(int c=0;c<G;++c){ float dd=l2_cpu(query+i*d,&cents[c*d],d); if(dd<bd){bd=dd;bc=c;} } asn[i]=bc; }
        std::vector<float> nx((size_t)G*d,0); std::vector<int> cnt(G,0);
        for(size_t i=0;i<M;++i){ int c=asn[i]; cnt[c]++; for(size_t j=0;j<d;++j) nx[c*d+j]+=query[i*d+j]; }
        for(int c=0;c<G;++c) if(cnt[c]) for(size_t j=0;j<d;++j) cents[c*d+j]=nx[c*d+j]/cnt[c];
    }
    std::vector<std::vector<int>> buckets(G);
    for(size_t i=0;i<M;++i) buckets[asn[i]].push_back((int)i);
    std::vector<std::vector<int>> gs;
    for(auto& bk:buckets) for(size_t s=0;s<bk.size();s+=B){ std::vector<int> g(bk.begin()+s, bk.begin()+std::min(bk.size(),s+B)); if(!g.empty()) gs.push_back(std::move(g)); }
    return gs;
}
// group_mode: 0 顺序 / 1 按主簇 / 2 按完整探查簇集合 / 3 query-KMeans
inline std::vector<std::vector<int>> make_groups(int group_mode, const std::vector<std::vector<int>>& probe,
                                                 const float* query, size_t M, size_t d, size_t B) {
    if (group_mode==0) return group_sequential(M,B);
    if (group_mode==1) return group_by_primary(probe,M,B);
    if (group_mode==2) return group_by_probeset(probe,M,B);
    return group_query_kmeans(query,M,d,B);
}

// 运行一个分组方案 (fine_mode 同 fine_group)，返回平均簇重叠率，填充 results
inline double run_groups(GpuCtx& g, const IVFIndex& idx, const PQCodebook& pq, int fine_mode,
                         const std::vector<std::vector<int>>& groups,
                         const std::vector<std::vector<int>>& probe, size_t k, int rerank_k,
                         const float* query_host, const float* rbase_host,
                         std::vector<std::vector<uint32_t>>& results) {
    results.assign(g.M, {});
    double tot_nc=0, tot_tiles=0;
    for (const auto& grp : groups) {
        double nc=0, tiles=0;
        fine_group(g, idx, pq, fine_mode, grp, probe, results, k, rerank_k, query_host, rbase_host, nc, tiles);
        tot_nc+=nc; tot_tiles+=tiles;
    }
    return (tot_tiles>0) ? (tot_nc/tot_tiles) : 0.0;
}
