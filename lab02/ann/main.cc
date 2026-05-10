#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include <queue>
#include <cmath>
#include <algorithm>
#include <arm_neon.h> 

#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"

using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if(!fin.is_open()) {
        std::cerr << "Failed to open " << data_path << std::endl;
        exit(-1);
    }
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(size_t i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();
    std::cerr<<"load data "<<data_path<<"\n";
    return data;
}

struct SearchResult {
    float recall;
    int64_t latency; 
};


//-----------------------------------------------------------------------------------------------------------------------
// 1. Flat SIMD
std::priority_queue<std::pair<float, uint32_t> > flat_search_simd(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    
    std::priority_queue<std::pair<float, uint32_t> > q;
    for(size_t i = 0; i < base_number; ++i) {
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        float* current_base = base + i * vecdim;
        for(size_t d = 0; d + 3 < vecdim; d += 4) {
            sum_vec = vfmaq_f32(sum_vec, vld1q_f32(current_base + d), vld1q_f32(query + d));
        }
        float dis = 1.0f - vaddvq_f32(sum_vec); 
        if(q.size() < k || dis < q.top().first) {
            q.push({dis, (uint32_t)i});
            if(q.size() > k) q.pop();
        }
    }
    return q;
}

//-----------------------------------------------------------------------------------------------------------------------

// 内积距离计算（其实和Flat-SIMD一样，这里怕混了单独提出来）
inline float calc_sub_dist(float* a, float* b, size_t dim) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    size_t d = 0;
    // 每次处理 4 个 float (128位)
    for (; d + 3 < dim; d += 4) {
        sum_vec = vfmaq_f32(sum_vec, vld1q_f32(a + d), vld1q_f32(b + d));
    }
    float sum = vaddvq_f32(sum_vec);
    // 处理不足 4 个的剩余维度
    for (; d < dim; ++d) {
        sum += a[d] * b[d];
    }
    return sum;
}

// 2. SQ SIMD
int8_t* sq_base = nullptr;
float sq_scale = 1.0f;
const size_t SQ_RERANK_CANDIDATES = 11;

void offline_sq(float* base, size_t base_number, size_t vecdim) {
    sq_base = new int8_t[base_number * vecdim];
    float max_val = 0;
    for (size_t i = 0; i < base_number * vecdim; ++i) max_val = std::max(max_val, std::abs(base[i]));
    sq_scale = 127.0f / max_val; // 对称量化
    #pragma omp parallel for
    for (size_t i = 0; i < base_number * vecdim; ++i) {
        sq_base[i] = (int8_t)(base[i] * sq_scale);
    }
}

std::priority_queue<std::pair<float, uint32_t> > sq_search_simd(
    int8_t* base_i8, 
    float* query, 
    float* base,       
    size_t base_number, 
    size_t vecdim, 
    size_t k
) {
    // SQ 量化向量 粗排
    std::priority_queue<std::pair<float, uint32_t> > coarse_q;
    int8_t q_i8[vecdim];
    for(size_t d = 0; d < vecdim; ++d) 
        q_i8[d] = (int8_t)(query[d] * sq_scale);

    for(size_t i = 0; i < base_number; ++i) {
        int8_t* cb = base_i8 + i * vecdim;
        int32x4_t sum_v = vdupq_n_s32(0);
        for(size_t d = 0; d + 15 < vecdim; d += 16) {
            int8x16_t b_v = vld1q_s8(cb + d);
            int8x16_t q_v = vld1q_s8(q_i8 + d);
            int16x8_t prod_l = vmull_s8(vget_low_s8(b_v), vget_low_s8(q_v));
            int16x8_t prod_h = vmull_s8(vget_high_s8(b_v), vget_high_s8(q_v));
            sum_v = vpadalq_s16(sum_v, prod_l);
            sum_v = vpadalq_s16(sum_v, prod_h);
        }
        float dis = 1.0f - (float)vaddvq_s32(sum_v) / (sq_scale * sq_scale);

        // 维护粗排候选集
        if(coarse_q.size() < SQ_RERANK_CANDIDATES || dis < coarse_q.top().first) {
            coarse_q.push({dis, (uint32_t)i});
            if(coarse_q.size() > SQ_RERANK_CANDIDATES)
                coarse_q.pop();
        }
    }

    // 原始向量 精排
    std::priority_queue<std::pair<float, uint32_t> > final_q;
    while(!coarse_q.empty()) {
        uint32_t idx = coarse_q.top().second;
        coarse_q.pop();
        float exact_ip = calc_sub_dist(query, base + idx * vecdim, vecdim);
        float exact_dis = 1.0f - exact_ip;

        // Top-K
        if(final_q.size() < k || exact_dis < final_q.top().first) {
            final_q.push({exact_dis, idx});
            if(final_q.size() > k)
                final_q.pop();
        }
    }

    return final_q;
}

//-----------------------------------------------------------------------------------------------------------------------
// 3. PQ-SIMD 
const int PQ_M = 16; // 参数调整 ** 
const int PQ_K = 256; 
uint8_t* pq_codes = nullptr;
float* pq_centroids = nullptr; 


void simple_kmeans(float* data, size_t n, size_t dim, float* cents) {
    for(int i=0; i<PQ_K; ++i) std::memcpy(cents + i*dim, data + (rand()%n)*dim, dim*sizeof(float));
    for(int iter=0; iter<50; ++iter) { // cishu
        std::vector<float> next_cents(PQ_K*dim, 0);
        std::vector<int> counts(PQ_K, 0);
        for(size_t i=0; i<n; ++i) {
            float max_ip = -1e9; int best = 0;
            for(int c=0; c<PQ_K; ++c) {
                float ip = calc_sub_dist(data+i*dim, cents+c*dim, dim);
                if(ip > max_ip) { max_ip = ip; best = c; }
            }
            counts[best]++;
            for(size_t d=0; d<dim; ++d) next_cents[best*dim+d] += data[i*dim+d];
        }
        for(int c=0; c<PQ_K; ++c) if(counts[c]) for(size_t d=0; d<dim; ++d) cents[c*dim+d] = next_cents[c*dim+d]/counts[c];
    }
}

void offline_pq(float* base, size_t base_number, size_t vecdim) {
    size_t sub_dim = vecdim / PQ_M;
    pq_codes = new uint8_t[base_number * PQ_M];
    pq_centroids = new float[PQ_M * PQ_K * sub_dim];
    for(int m=0; m<PQ_M; ++m) {
        std::vector<float> sub_data(base_number * sub_dim);
        for(size_t i=0; i<base_number; ++i) std::memcpy(&sub_data[i*sub_dim], base+i*vecdim+m*sub_dim, sub_dim*sizeof(float));
        simple_kmeans(sub_data.data(), base_number, sub_dim, pq_centroids + m*PQ_K*sub_dim);
        #pragma omp parallel for
        for(size_t i=0; i<base_number; ++i) {
            float max_ip = -1e9; int best = 0;
            for(int c=0; c<PQ_K; ++c) {
                float ip = calc_sub_dist(base+i*vecdim+m*sub_dim, pq_centroids+m*PQ_K*sub_dim+c*sub_dim, sub_dim);
                if(ip > max_ip) { max_ip = ip; best = c; }
            }
            pq_codes[i*PQ_M + m] = best;
        }
    }
}

std::priority_queue<std::pair<float, uint32_t> > pq_search_simd(float* query, size_t base_number, size_t vecdim, size_t k) {
    

    size_t sub_dim = vecdim / PQ_M;
    float lut[PQ_M][PQ_K];
    for(int m=0; m<PQ_M; ++m) 
        for(int c=0; c<PQ_K; ++c) 
            lut[m][c] = calc_sub_dist(query+m*sub_dim, pq_centroids+m*PQ_K*sub_dim+c*sub_dim, sub_dim);

    std::priority_queue<std::pair<float, uint32_t> > q;
    for(size_t i = 0; i < base_number; ++i) {
        float ip = 0;
        uint8_t* code = pq_codes + i*PQ_M;
        for(int m=0; m<PQ_M; ++m) ip += lut[m][code[m]];
        float dis = 1.0f - ip;
        if(q.size() < k || dis < q.top().first) {
            q.push({dis, (uint32_t)i});
            if(q.size() > k) q.pop();
        }
    }
    return q;
}


//-----------------------------------------------------------------------------------------------------------------------
// 4. PQ + Re-ranking SIMD

void offline_pq_ip(float* base, size_t base_number, size_t vecdim) {
    size_t sub_dim = vecdim / PQ_M;
    pq_codes = new uint8_t[base_number * PQ_M];
    pq_centroids = new float[PQ_M * PQ_K * sub_dim];
    
    // 随机采样 20000 条数据训练 KMeans 加快速度
    size_t train_n = std::min((size_t)20000, base_number); 
    
    for(int m=0; m<PQ_M; ++m) {
        std::vector<float> sub_data(train_n * sub_dim);
        for(size_t i=0; i<train_n; ++i) 
            std::memcpy(&sub_data[i*sub_dim], base+i*vecdim+m*sub_dim, sub_dim*sizeof(float));
            
        simple_kmeans(sub_data.data(), train_n, sub_dim, pq_centroids + m*PQ_K*sub_dim);
        
        #pragma omp parallel for // OpenMP 
        for(size_t i=0; i<base_number; ++i) {
            for(int m=0; m<PQ_M; ++m) {
                float max_ip = -1e9;
                int best = 0;
                float* sub_vec = base + i * vecdim + m * sub_dim;
                float* centers = pq_centroids + m * PQ_K * sub_dim;
                
                for(int c=0; c<PQ_K; ++c) {
                    float ip = calc_sub_dist(sub_vec, centers + c * sub_dim, sub_dim);
                    if(ip > max_ip) { max_ip = ip; best = c; }
                }
                pq_codes[i * PQ_M + m] = (uint8_t)best;
            }
        }
    }
}

std::priority_queue<std::pair<float, uint32_t> > pq_rerank_search_simd(
    float* query, float* base, size_t base_number, size_t vecdim, size_t k, size_t rerank_k = 500) {
    
    
    size_t sub_dim = vecdim / PQ_M;
    float lut[PQ_M][PQ_K];
    
    // 构建 LUT
    for(int m=0; m<PQ_M; ++m) {
        for(int c=0; c<PQ_K; ++c) {
            lut[m][c] = calc_sub_dist(query+m*sub_dim, pq_centroids+m*PQ_K*sub_dim+c*sub_dim, sub_dim);
        }
    }

    // ADC 粗排：找出前 rerank_k 个候选者
    std::priority_queue<std::pair<float, uint32_t> > coarse_q;
    for(size_t i = 0; i < base_number; ++i) {
        float ip = 0;
        uint8_t* code = pq_codes + i*PQ_M;
        for(int m=0; m<PQ_M; ++m) ip += lut[m][code[m]];
        
        float dis = 1.0f - ip;
        if(coarse_q.size() < rerank_k || dis < coarse_q.top().first) {
            coarse_q.push({dis, (uint32_t)i});
            if(coarse_q.size() > rerank_k) coarse_q.pop();
        }
    }

    // 精排
    std::priority_queue<std::pair<float, uint32_t> > fine_q;
    while(!coarse_q.empty()) {
        uint32_t idx = coarse_q.top().second;
        coarse_q.pop();

        // 提取原始浮点向量
        float exact_ip = calc_sub_dist(query, base + idx * vecdim, vecdim);
        float exact_dis = 1.0f - exact_ip;

        // 压入最终的 Top-k 队列
        if(fine_q.size() < k || exact_dis < fine_q.top().first) {
            fine_q.push({exact_dis, idx});
            if(fine_q.size() > k) fine_q.pop();
        }
    }
    
    return fine_q;
}


//-----------------------------------------------------------------------------------------------------------------------
// 5：PQ-Final

void offline_pq_ip_5(float* base, size_t base_number, size_t vecdim) {
    size_t sub_dim = vecdim / PQ_M;
    pq_codes = new uint8_t[base_number * PQ_M];
    pq_centroids = new float[PQ_M * PQ_K * sub_dim];
    
    size_t train_n = std::min((size_t)20000, base_number); 
    for(int m=0; m<PQ_M; ++m) {
        std::vector<float> sub_data(train_n * sub_dim);
        for(size_t i=0; i<train_n; ++i) 
            std::memcpy(&sub_data[i*sub_dim], base+i*vecdim+m*sub_dim, sub_dim*sizeof(float));
        simple_kmeans(sub_data.data(), train_n, sub_dim, pq_centroids + m*PQ_K*sub_dim);
    }

    #pragma omp parallel for schedule(dynamic, 256)
    for(size_t i=0; i<base_number; ++i) {
        for(int m=0; m<PQ_M; ++m) {
            float max_ip = -1e9; int best = 0;
            float* sub_vec = base + i * vecdim + m * sub_dim;
            float* centers = pq_centroids + m * PQ_K * sub_dim;
            for(int c=0; c<PQ_K; ++c) {
                float ip = calc_sub_dist(sub_vec, centers + c * sub_dim, sub_dim);
                if(ip > max_ip) { max_ip = ip; best = c; }
            }
            pq_codes[i * PQ_M + m] = (uint8_t)best;
        }
    }
}
// SoA 布局编码表 [PQ_M][base_number]
uint8_t* pq_codes_soa = nullptr;

void prepare_soa_codes(size_t base_number) {
    pq_codes_soa = new uint8_t[PQ_M * base_number];
    for (int m = 0; m < PQ_M; ++m) {
        for (size_t i = 0; i < base_number; ++i) {
            pq_codes_soa[m * base_number + i] = pq_codes[i * PQ_M + m];
        }
    }
}

float* pq_centroids_soa = nullptr; // 维度：[PQ_M][sub_dim][PQ_K]

void prepare_soa_centroids(size_t vecdim) {
    size_t sub_dim = vecdim / PQ_M;
    pq_centroids_soa = new float[PQ_M * sub_dim * PQ_K];
    for (int m = 0; m < PQ_M; ++m) {
        for (int d = 0; d < (int)sub_dim; ++d) {
            for (int c = 0; c < PQ_K; ++c) {
                pq_centroids_soa[m * sub_dim * PQ_K + d * PQ_K + c] = 
                    pq_centroids[m * PQ_K * sub_dim + c * sub_dim + d];
            }
        }
    }
}

// 局部精排
std::priority_queue<std::pair<float, uint32_t>> pq_search_final(
    float* query, float* base, size_t base_number, size_t vecdim, size_t k, size_t rerank_k) {
    

    size_t sub_dim = vecdim / PQ_M;
    alignas(64) float lut[PQ_M][PQ_K];

    // LUT 构建
    for (int m = 0; m < PQ_M; ++m) {
        float* sub_q = query + m * sub_dim;
        float* m_soa_cent = pq_centroids_soa + m * sub_dim * PQ_K;
        for (int c = 0; c < PQ_K; c += 4) {
            float32x4_t acc_ip = vdupq_n_f32(0.0f);
            for (size_t d = 0; d < sub_dim; ++d) {
                acc_ip = vfmaq_f32(acc_ip, vdupq_n_f32(sub_q[d]), vld1q_f32(m_soa_cent + d * PQ_K + c));
            }
            vst1q_f32(&lut[m][c], acc_ip);
        }
    }

    // 每个线程给出自己的 Top-K
    int max_threads = omp_get_max_threads();
    std::vector<std::vector<std::pair<float, uint32_t>>> thread_results(max_threads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        // 线程局部粗排
        std::priority_queue<std::pair<float, uint32_t>> local_coarse_q;
        
        #pragma omp for schedule(static)
        for (size_t i = 0; i < base_number; ++i) {
            float ip = 0;
            /*
            // 循环展开
            ip += lut[0][pq_codes_soa[0 * base_number + i]];
            ip += lut[1][pq_codes_soa[1 * base_number + i]];
            ip += lut[2][pq_codes_soa[2 * base_number + i]];
            ip += lut[3][pq_codes_soa[3 * base_number + i]];
            ip += lut[4][pq_codes_soa[4 * base_number + i]];
            ip += lut[5][pq_codes_soa[5 * base_number + i]];
            ip += lut[6][pq_codes_soa[6 * base_number + i]];
            ip += lut[7][pq_codes_soa[7 * base_number + i]];
            */
            
            for (int m = 0; m < PQ_M; ++m) {
                ip += lut[m][pq_codes_soa[m * base_number + i]];
            }

            float dis = 1.0f - ip;
            if (local_coarse_q.size() < rerank_k || dis < local_coarse_q.top().first) {
                local_coarse_q.push({dis, (uint32_t)i});
                if (local_coarse_q.size() > rerank_k) local_coarse_q.pop();
            }
        }

        // 线程内部直接进行精排
        std::priority_queue<std::pair<float, uint32_t>> local_fine_q;
        while (!local_coarse_q.empty()) {
            uint32_t idx = local_coarse_q.top().second;
            local_coarse_q.pop();
            
            float exact_dis = 1.0f - calc_sub_dist(query, base + idx * vecdim, vecdim);
            if (local_fine_q.size() < k || exact_dis < local_fine_q.top().first) {
                local_fine_q.push({exact_dis, idx});
                if (local_fine_q.size() > k) local_fine_q.pop();
            }
        }

        // 将该线程的最终 top-k 存入结果集
        while (!local_fine_q.empty()) {
            thread_results[tid].push_back(local_fine_q.top());
            local_fine_q.pop();
        }
    }

    // 归并（此时只有 num_threads * k 条数据）
    std::priority_queue<std::pair<float, uint32_t>> final_q;
    for (const auto& tr : thread_results) {
        for (const auto& res : tr) {
            if (final_q.size() < k || res.first < final_q.top().first) {
                final_q.push(res);
                if (final_q.size() > k) final_q.pop();
            }
        }
    }
    return final_q;
}


//-----------------------------------------------------------------------------------------------------------------------


int main(int argc, char *argv[]) {
    size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    test_number = 2000;
    const size_t k = 10;
    std::vector<SearchResult> results(test_number);

    int strategy = 5; // 对比不同算法
                      // 1 = Flat-SIMD, 2 = SQ-SIMD, 
                      // 3 = PQ-SIMD, 4 = PQ-rerank-SIMD, 5 = PQ-final
    int rerank_k = 125;

    std::cout << "Running Strategy: " << strategy << std::endl;

    if (strategy == 2) offline_sq(base, base_number, vecdim);
    if (strategy == 3) offline_pq(base, base_number, vecdim);
    if (strategy == 4) offline_pq_ip(base, base_number, vecdim);
    if (strategy == 5){
        offline_pq_ip_5(base, base_number, vecdim);
        prepare_soa_centroids(vecdim);
        prepare_soa_codes(base_number);
    } 

    for(size_t i = 0; i < test_number; ++i) {
        struct timeval start, end;
        gettimeofday(&start, NULL);

        std::priority_queue<std::pair<float, uint32_t>> res;
        if (strategy == 1) res = flat_search_simd(base, test_query + i*vecdim, base_number, vecdim, k);
        else if (strategy == 2) res = sq_search_simd(sq_base, test_query + i*vecdim,base, base_number, vecdim, k);
        else if (strategy == 3) res = pq_search_simd(test_query + i*vecdim, base_number, vecdim, k);
        else if (strategy == 4) res = pq_rerank_search_simd(test_query + i*vecdim, base, base_number, vecdim, k, rerank_k); // 500 是 rerank_k，你可以调整它来观察 Latency 和 Recall 的权衡
        else if (strategy == 5) res = pq_search_final(test_query + i*vecdim, base, base_number, vecdim, k, rerank_k);
        gettimeofday(&end, NULL);
        int64_t diff = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);

        std::set<uint32_t> gtset;
        for(size_t j = 0; j < k; ++j) gtset.insert(test_gt[j + i*test_gt_d]);

        size_t acc = 0;
        while (!res.empty()) {
            if(gtset.count(res.top().second)) ++acc;
            res.pop();
        }
        results[i] = {(float)acc/k, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for(auto& r : results) { avg_recall += r.recall; avg_latency += r.latency; }
    std::cout << "Avg Recall: " << avg_recall / test_number << "\nAvg Latency: " << avg_latency / test_number << " us\n";


    if (strategy == 3) {
        std::cout << "PQ_M = " << PQ_M << "\n";
    }
    else if (strategy == 4) {
        std::cout << "PQ_M = " << PQ_M << "\n";
        std::cout << "rerank_k = " << rerank_k << "\n";    
    }
    else if (strategy == 5) {
        std::cout << "PQ_M = " << PQ_M << "\n";
        std::cout << "rerank_k = " << rerank_k << "\n";    
    }

    std::cout << "\n" << "====================" << "\n";


    delete[] test_query; delete[] test_gt; delete[] base;
    if(sq_base) delete[] sq_base; if(pq_codes) delete[] pq_codes; if(pq_centroids) delete[] pq_centroids;
    return 0;
}
