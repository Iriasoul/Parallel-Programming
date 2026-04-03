// 并行程序设计Lab1 实验1：矩阵与向量内积
#include <iostream>
#include <string>
#include <windows.h>
#include <fstream>
#include <cstdlib>

using namespace std;

// Windows高精度计时
LARGE_INTEGER get_counter() {
    LARGE_INTEGER cnt;
    QueryPerformanceCounter(&cnt);
    return cnt;
}

double get_time(LARGE_INTEGER s, LARGE_INTEGER e, LARGE_INTEGER freq) {
    return (e.QuadPart - s.QuadPart) * 1000000.0 / freq.QuadPart;
}

// 平凡算法
void plain(int n, double** b, double* a, double* sum) {
    int i, j;
    for (i = 0; i < n; i++) {
        sum[i] = 0.0;
        for (j = 0; j < n; j++) {
            sum[i] += b[j][i] * a[j];
        }
    }
}

// Cache优化算法
void cache_opt(int n, double** b, double* a, double* sum) {
    int i, j;
    for (i = 0; i < n; i++) {
        sum[i] = 0.0;
    }
    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            sum[i] += b[j][i] * a[j];
        }
    }
}

// Cache优化 + 4路循环展开
void cache_unroll(int n, double** b, double* a, double* sum) {
    int i, j;
    for (i = 0; i < n; i++) {
        sum[i] = 0.0;
    }
    for (j = 0; j < n; j++) {
        // 4路循环展开，降低循环开销
        for (i = 0; i + 3 < n; i += 4) {
            sum[i] += b[j][i] * a[j];
            sum[i+1] += b[j][i+1] * a[j];
            sum[i+2] += b[j][i+2] * a[j];
            sum[i+3] += b[j][i+3] * a[j];
        }
        // 处理剩余元素
        for (; i < n; i++) {
            sum[i] += b[j][i] * a[j];
        }
    }
}

// 数据初始化
void init_data(int n, double** b, double* a) {
    int i, j;
    for (i = 0; i < n; i++) {
        a[i] = 1.0;
        for (j = 0; j < n; j++) {
            b[i][j] = (double)(i + j);
        }
    }
}

int main() {
    int sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192};
    int test_num = sizeof(sizes) / sizeof(int);
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    ofstream out("mmv_result.txt");
    if (!out.is_open()) {
        cout << "文件打开失败" << endl;
        return 1;
    }

    int idx;
    for (idx = 0; idx < test_num; idx++) {
        int n = sizes[idx];
        int i, j;

        // 动态分配内存
        double** b = new double*[n];
        for (i = 0; i < n; i++) {
            b[i] = new double[n];
        }
        double* a = new double[n];
        double* sum = new double[n];

        init_data(n, b, a);

        // 平凡算法
        LARGE_INTEGER s1 = get_counter();
        for (i = 0; i < 100; i++) plain(n, b, a, sum);
        LARGE_INTEGER e1 = get_counter();
        double t_plain = get_time(s1, e1, freq) / 100.0;

        // Cache优化算法
        LARGE_INTEGER s2 = get_counter();
        for (i = 0; i < 100; i++) cache_opt(n, b, a, sum);
        LARGE_INTEGER e2 = get_counter();
        double t_cache = get_time(s2, e2, freq) / 100.0;

        // 循环展开算法
        LARGE_INTEGER s3 = get_counter();
        for (i = 0; i < 100; i++) cache_unroll(n, b, a, sum);
        LARGE_INTEGER e3 = get_counter();
        double t_unroll = get_time(s3, e3, freq) / 100.0;

        // 输出结果
        out << "N = " << n << endl;
        out << "平凡算法: " << t_plain << " us" << endl;
        out << "Cache优化: " << t_cache << " us" << endl;
        out << "循环展开: " << t_unroll << " us" << endl;
        out << "-------------------------" << endl;

        // 释放内存
        for (i = 0; i < n; i++) delete[] b[i];
        delete[] b;
        delete[] a;
        delete[] sum;
    }

    out.close();
    cout << "矩阵向量乘实验完成，结果已保存至 mmv_result.txt" << endl;
    return 0;
}
