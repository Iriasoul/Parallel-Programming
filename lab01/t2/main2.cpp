// 并行程序设计Lab1 实验2：n个数求和
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
void sum_plain(int n, double* a, double& sum) {
    int i;
    sum = 0.0;
    for (i = 0; i < n; i++) {
        sum += a[i];
    }
}

// 两路优化算法
void sum_opt(int n, double* a, double& sum) {
    int i;
    double sum1 = 0.0, sum2 = 0.0;
    for (i = 0; i < n; i += 2) {
        sum1 += a[i];
        sum2 += a[i + 1];
    }
    sum = sum1 + sum2;
}

// 4路循环展开
void sum_unroll(int n, double* a, double& sum) {
    int i;
    double sum1 = 0.0, sum2 = 0.0;
    double sum3 = 0.0, sum4 = 0.0;
    for (i = 0; i + 3 < n; i += 4) {
        sum1 += a[i];
        sum2 += a[i+1];
        sum3 += a[i+2];
        sum4 += a[i+3];
    }
    // 处理剩余元素
    for (; i < n; i++) {
        sum1 += a[i];
    }
    sum = sum1 + sum2 + sum3 + sum4;
}

// 数据初始化
void init_arr(int n, double* a) {
    int i;
    for (i = 0; i < n; i++) {
        a[i] = 1.0;
    }
}

int main() {
    int sizes[] = {1024, 2048, 4096, 32768, 1048576, 33554432};
    int test_num = sizeof(sizes) / sizeof(int);
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    ofstream out("sum_result.txt");
    if (!out.is_open()) {
        cout << "文件打开失败" << endl;
        return 1;
    }

    int idx;
    for (idx = 0; idx < test_num; idx++) {
        int n = sizes[idx];
        int i;

        double* a = new double[n];
        double res;

        init_arr(n, a);

        // 平凡算法
        LARGE_INTEGER s1 = get_counter();
        for (i = 0; i < 1000; i++) sum_plain(n, a, res);
        LARGE_INTEGER e1 = get_counter();
        double t_plain = get_time(s1, e1, freq) / 1000.0;

        // 两路优化算法
        LARGE_INTEGER s2 = get_counter();
        for (i = 0; i < 1000; i++) sum_opt(n, a, res);
        LARGE_INTEGER e2 = get_counter();
        double t_opt = get_time(s2, e2, freq) / 1000.0;

        // 4路循环展开
        LARGE_INTEGER s3 = get_counter();
        for (i = 0; i < 1000; i++) sum_unroll(n, a, res);
        LARGE_INTEGER e3 = get_counter();
        double t_unroll = get_time(s3, e3, freq) / 1000.0;

        // 输出结果
        out << "N = " << n << endl;
        out << "平凡算法: " << t_plain << " us" << endl;
        out << "两路优化: " << t_opt << " us" << endl;
        out << "4路循环展开: " << t_unroll << " us" << endl;
        out << "-------------------------" << endl;

        delete[] a;
    }

    out.close();
    cout << "数组求和实验完成，结果已保存至 sum_result.txt" << endl;
    return 0;
}
