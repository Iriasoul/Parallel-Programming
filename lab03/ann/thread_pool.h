#pragma once
#include <pthread.h>
#include <functional>
#include <vector>

struct ThreadPool {
    struct WorkerArg { ThreadPool* pool; int tid; };

    int  n        = 0;
    int  work_ver = 0;      // 版本号, 每次 dispatch 递增; worker 凭此判断有无新任务
    int  pending  = 0;      // 仍在运行的 worker 数量
    bool shutdown = false;
    std::function<void(int)> fn;

    pthread_mutex_t mu;
    pthread_cond_t  cv_work;   // worker 在此等待新任务
    pthread_cond_t  cv_done;   // master 在此等待所有 worker 完成

    std::vector<pthread_t>  threads;
    std::vector<WorkerArg>  args;

    static void* worker_loop(void* raw) {
        WorkerArg* wa = (WorkerArg*)raw;
        ThreadPool* p = wa->pool;
        int tid = wa->tid, seen = 0;
        while (true) {
            pthread_mutex_lock(&p->mu);
            // 等待直到有新任务 (work_ver 变化) 或 shutdown 信号
            while (p->work_ver == seen && !p->shutdown)
                pthread_cond_wait(&p->cv_work, &p->mu);
            if (p->shutdown) { pthread_mutex_unlock(&p->mu); return nullptr; }
            seen = p->work_ver;
            auto f = p->fn;                     // 复制 fn (shared_ptr 语义, 开销极小)
            pthread_mutex_unlock(&p->mu);        // 释放锁后再执行, 不阻塞其他 worker

            f(tid);                              // 执行本次任务

            pthread_mutex_lock(&p->mu);
            if (--p->pending == 0)
                pthread_cond_signal(&p->cv_done); // 最后一个完成的 worker 通知 master
            pthread_mutex_unlock(&p->mu);
        }
    }

    // 创建 num 个工作线程并让它们进入等待状态
    void init(int num) {
        n = num;
        pthread_mutex_init(&mu,      nullptr);
        pthread_cond_init(&cv_work,  nullptr);
        pthread_cond_init(&cv_done,  nullptr);
        threads.resize(n);
        args.resize(n);
        for (int i = 0; i < n; ++i) {
            args[i] = {this, i};
            pthread_create(&threads[i], nullptr, worker_loop, &args[i]);
        }
    }

    // 向所有 n 个 worker 分发任务 fn(tid), 阻塞直到全部完成
    void dispatch(std::function<void(int)> work) {
        pthread_mutex_lock(&mu);
        fn       = work;
        pending  = n;
        work_ver++;
        pthread_cond_broadcast(&cv_work);           // 唤醒所有 worker
        while (pending > 0)
            pthread_cond_wait(&cv_done, &mu);        // 等待全部完成
        pthread_mutex_unlock(&mu);
    }

    // 通知所有 worker 退出, 然后 join
    void destroy() {
        pthread_mutex_lock(&mu);
        shutdown = true;
        pthread_cond_broadcast(&cv_work);
        pthread_mutex_unlock(&mu);
        for (auto& t : threads) pthread_join(t, nullptr);
        pthread_mutex_destroy(&mu);
        pthread_cond_destroy(&cv_work);
        pthread_cond_destroy(&cv_done);
    }
};
