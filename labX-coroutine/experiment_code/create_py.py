"""测量创建一个并发单元的平均开销 (创建+启动+结束)。"""
import time, threading, asyncio, json

def thread_create(n=2000):
    t0 = time.perf_counter()
    ts = [threading.Thread(target=lambda: None) for _ in range(n)]
    for t in ts: t.start()
    for t in ts: t.join()
    return (time.perf_counter()-t0)/n*1e6   # us/unit

def asyncio_create(n=100000):
    async def main():
        async def noop(): return
        t0 = time.perf_counter()
        await asyncio.gather(*[noop() for _ in range(n)])
        return (time.perf_counter()-t0)/n*1e6
    return asyncio.run(main())

if __name__ == "__main__":
    print(json.dumps({
        "thread_us_per_create": round(thread_create(),3),
        "asyncio_us_per_create": round(asyncio_create(),3),
    }))
