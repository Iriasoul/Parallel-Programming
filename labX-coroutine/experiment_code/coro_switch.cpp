// C++20 无栈协程: 测量一次 resume+suspend 周期的延迟
#include <coroutine>
#include <cstdio>
#include <chrono>

struct Task {
    struct promise_type {
        Task get_return_object() { return Task{std::coroutine_handle<promise_type>::from_promise(*this)}; }
        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend()   noexcept { return {}; }
        void return_void() {}
        void unhandled_exception() {}
    };
    std::coroutine_handle<promise_type> h;
};

// 协程: 反复挂起, 由主循环恢复
Task pingpong(volatile long* counter) {
    while (true) {
        ++(*counter);
        co_await std::suspend_always{};   // 挂起, 交回控制权
    }
}

int main() {
    const long ITERS = 50'000'000;
    volatile long counter = 0;
    Task t = pingpong(&counter);
    auto h = t.h;

    auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < ITERS; ++i) {
        h.resume();                       // 恢复 -> 协程跑到下一个 co_await 再交回
    }
    auto t1 = std::chrono::steady_clock::now();
    h.destroy();

    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / ITERS;
    printf("{\"mode\":\"cpp_coroutine_switch\",\"iters\":%ld,\"ns_per_switch\":%.2f}\n", ITERS, ns);
    return 0;
}
