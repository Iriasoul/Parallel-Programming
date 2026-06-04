# 协程实验代码

测试环境: 1 vCPU Intel Xeon @2.8GHz, 3.9GB RAM, Ubuntu 24.04
Python 3.12 / Go 1.22 / g++ 13.3 (C++20)

## 文件说明
- mem_py.py        : 测量 Python OS 线程 与 asyncio 协程 的单位内存占用 (RSS)
- mem_go.go        : 测量 Go goroutine 的单位内存占用 (扩展到 10 万 goroutine)
- create_py.py     : 测量 Python 线程 / asyncio 任务 的创建耗时
- create_go.go     : 测量 Go goroutine 的创建耗时 (生成 100 万)
- io_py.py         : I/O 吞吐对比 (线程池 50/200/1000 vs asyncio 单线程, N=2000, 每请求 sleep 20ms)
- coro_switch.cpp  : C++20 无栈协程 resume/suspend 切换延迟
- thread_switch.cpp: OS 线程 condvar 乒乓 上下文切换延迟
- results.json     : 全部实测结果汇总

## 复现方法
python3 mem_py.py
go run mem_go.go            # 需要 GOCACHE=/tmp/gocache
python3 create_py.py
go run create_go.go
python3 io_py.py
g++ -std=c++20 -O2 coro_switch.cpp -o coro_switch && ./coro_switch
g++ -std=c++20 -O2 -pthread thread_switch.cpp -o thread_switch && ./thread_switch
