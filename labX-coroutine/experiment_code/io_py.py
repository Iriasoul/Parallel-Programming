"""模拟服务器处理 N 个并发请求, 每个请求阻塞 D 秒模拟网络/磁盘 I/O。
对比: 固定大小线程池(生产常见模式) vs 单线程 asyncio 事件循环。
输出总耗时与吞吐(req/s)。"""
import sys, time, asyncio, json
from concurrent.futures import ThreadPoolExecutor

N = 2000          # 总请求数
D = 0.020         # 每请求 I/O 等待 20ms

def io_task():
    time.sleep(D)             # 阻塞式 I/O (线程在此让出 GIL)
    return 1

def run_threadpool(workers):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda _: io_task(), range(N)))
    dt = time.perf_counter() - t0
    return dt

async def io_coro():
    await asyncio.sleep(D)    # 非阻塞 I/O, 协程在此挂起
    return 1

def run_asyncio():
    async def main():
        t0 = time.perf_counter()
        await asyncio.gather(*[io_coro() for _ in range(N)])
        return time.perf_counter() - t0
    return asyncio.run(main())

if __name__ == "__main__":
    results = {}
    for w in [50, 200, 1000]:
        dt = run_threadpool(w)
        results[f"threadpool_{w}"] = {"ms": round(dt*1000,1), "rps": round(N/dt)}
    dt = run_asyncio()
    results["asyncio"] = {"ms": round(dt*1000,1), "rps": round(N/dt)}
    print(json.dumps(results, indent=2))
