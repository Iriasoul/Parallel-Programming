"""测量 N 个并发单元(线程 vs asyncio协程)处于空闲挂起态时的进程 RSS。
每个单元都阻塞在同步原语上, 保证同时存活。输出每单元净增内存(KB)。"""
import sys, threading, asyncio, os, json

def rss_kb():
    with open(f"/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1

def bench_threads(n):
    ev = threading.Event()
    threads = []
    base = rss_kb()
    def worker():
        ev.wait()                 # 全部线程在此挂起
    for _ in range(n):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
    peak = rss_kb()
    ev.set()
    for t in threads:
        t.join()
    return (peak - base) / n      # 每线程净增 KB

def bench_asyncio(n):
    async def main():
        base = rss_kb()
        ev = asyncio.Event()
        async def coro():
            await ev.wait()       # 全部协程在此挂起
        tasks = [asyncio.create_task(coro()) for _ in range(n)]
        await asyncio.sleep(0.2)  # 让事件循环把所有任务挂起
        peak = rss_kb()
        ev.set()
        await asyncio.gather(*tasks)
        return (peak - base) / n
    return asyncio.run(main())

if __name__ == "__main__":
    mode = sys.argv[1]; n = int(sys.argv[2])
    if mode == "thread":
        print(json.dumps({"mode":"thread","n":n,"kb_per_unit":bench_threads(n)}))
    else:
        print(json.dumps({"mode":"asyncio","n":n,"kb_per_unit":bench_asyncio(n)}))
