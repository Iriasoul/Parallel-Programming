// 两个 OS 线程互相唤醒(ping-pong), 测量一次内核态线程切换的延迟
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstdio>
#include <chrono>

int turn = 0;               // 0 => A 跑, 1 => B 跑
std::mutex m;
std::condition_variable cv;
const long ITERS = 200'000; // 线程切换很贵, 用较少迭代

void worker(int id) {
    for (long i = 0; i < ITERS; ++i) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&]{ return turn == id; });
        turn = 1 - id;        // 让对方跑
        cv.notify_one();      // 唤醒对方 -> 触发线程切换
    }
}

int main() {
    auto t0 = std::chrono::steady_clock::now();
    std::thread a(worker, 0), b(worker, 1);
    a.join(); b.join();
    auto t1 = std::chrono::steady_clock::now();

    // 总切换次数约为 2*ITERS
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / (2.0 * ITERS);
    printf("{\"mode\":\"thread_ctxswitch\",\"switches\":%ld,\"ns_per_switch\":%.2f}\n", 2*ITERS, ns);
    return 0;
}
