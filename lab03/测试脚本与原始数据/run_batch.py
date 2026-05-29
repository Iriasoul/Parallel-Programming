#!/usr/bin/env python3
"""
Lab3 批量性能测试脚本
用法:
  python3 run_batch.py              # 全部测试组
  python3 run_batch.py flat         # 只跑 flat 组
  python3 run_batch.py flat pq ivf  # 多个组

可用组名: flat  pq  ivf  ivfpq  hnsw  threads  all(默认)
输出: results/YYYYMMDD_HHMMSS/all_results.csv
"""

import subprocess, sys, os, re, csv
from datetime import datetime

BINARY  = "./main"
REPEAT  = 10        # 每组重复测量次数，取延迟最小值（recall 取第一次即可）
OUTDIR  = f"results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTDIR, exist_ok=True)

CSV_PATH = os.path.join(OUTDIR, "all_results.csv")
_csv_file = open(CSV_PATH, "w", newline="")
_writer   = csv.writer(_csv_file)
_writer.writerow(["group","strategy","rerank_k","nprobe","nlist","threads","ef",
                  "recall","latency_us"])

def run(group, strategy, **kwargs):
    """执行一次测试（重复 REPEAT 次，取延迟最小值），将结果写入 CSV 并打印到终端。"""
    # 构造命令行参数
    cmd = [BINARY, "--strategy", str(strategy)]
    for k, v in kwargs.items():
        cmd += [f"--{k}", str(v)]

    # 提取各参数值（未指定时填 -）
    rk  = kwargs.get("rerank_k", "-")
    np_ = kwargs.get("nprobe",   "-")
    nl  = kwargs.get("nlist",    256)
    t   = kwargs.get("omp", kwargs.get("pthreads", 4))
    ef  = kwargs.get("ef",       64)

    extra = " ".join(f"--{k} {v}" for k, v in kwargs.items())
    print(f"  [S{strategy:2}] {group:<8} {extra:<35}", end="", flush=True)

    recall  = "ERROR"
    latency = "ERROR"
    latencies = []

    for i in range(REPEAT):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            out = result.stdout
        except FileNotFoundError:
            print("FAILED (binary not found)")
            _writer.writerow([group,strategy,rk,np_,nl,t,ef,"ERROR","ERROR"])
            return
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            _writer.writerow([group,strategy,rk,np_,nl,t,ef,"TIMEOUT","TIMEOUT"])
            return

        m_recall  = re.search(r"Avg Recall:\s+([\d.]+)",  out)
        m_latency = re.search(r"Avg Latency:\s+([\d.]+)", out)

        if m_recall and m_latency:
            if i == 0:
                recall = m_recall.group(1)   # recall 取第一次（确定性结果）
            latencies.append(float(m_latency.group(1)))
        else:
            print(f"PARSE_ERR  stdout={out[:80]!r}")
            _writer.writerow([group,strategy,rk,np_,nl,t,ef,"PARSE_ERR","PARSE_ERR"])
            return

        # 追加原始日志（每次都记录）
        log = os.path.join(OUTDIR, f"{group}.log")
        with open(log, "a") as f:
            f.write(f"\n=== S{strategy} {extra} [run {i+1}/{REPEAT}] ===\n{out}\n")

    latency = f"{min(latencies):.2f}"
    all_lat = " / ".join(f"{x:.1f}" for x in latencies)
    print(f"recall={recall:<8}  latency={latency} us  (all: {all_lat})")

    _writer.writerow([group,strategy,rk,np_,nl,t,ef,recall,latency])
    _csv_file.flush()


# ---------------------------------------------------------------------------
# 各测试组
# ---------------------------------------------------------------------------

def run_flat():
    print("[flat] Flat-SIMD 单线程 vs 多线程加速比")
    run("flat", 0)
    for t in [1, 2, 4, 8]:
        run("flat", 1, omp=t)
        run("flat", 2, pthreads=t)

def run_pq():
    print("[pq] PQ baseline 参数扫描 & 多线程加速比")
    for rk in [50, 100, 200, 500, 1000]:
        run("pq", 3, rerank_k=rk)
        run("pq", 4, rerank_k=rk)
    for t in [1, 2, 4, 8]:
        run("pq", 5, rerank_k=500, omp=t)
        run("pq", 6, rerank_k=500, pthreads=t)

def run_ivf():
    print("[ivf] IVF-SIMD 参数扫描 & 多线程加速比")
    for np in [4, 8, 16, 32, 64]:
        run("ivf", 7, nprobe=np)
    for t in [1, 2, 4, 8]:
        run("ivf", 8, nprobe=16, omp=t)
        run("ivf", 9, nprobe=16, pthreads=t)
    for np in [4, 8, 16, 32, 64]:
        run("ivf", 8, nprobe=np, omp=4)
        run("ivf", 9, nprobe=np, pthreads=4)

def run_ivfpq():
    print("[ivfpq] IVF-PQ 两种方法对比")
    for np in [4, 8, 16, 32, 64]:
        run("ivfpq", 10, nprobe=np, rerank_k=500)
        run("ivfpq", 13, nprobe=np, rerank_k=500)
    for t in [1, 2, 4, 8]:
        run("ivfpq", 11, nprobe=16, rerank_k=500, omp=t)
        run("ivfpq", 12, nprobe=16, rerank_k=500, pthreads=t)
        run("ivfpq", 14, nprobe=16, rerank_k=500, omp=t)
        run("ivfpq", 15, nprobe=16, rerank_k=500, pthreads=t)

def run_hnsw():
    print("[hnsw] HNSW 参数扫描 & 多线程对比")
    for ef in [16, 32, 64, 128, 200]:
        run("hnsw", 16, ef=ef)
        run("hnsw", 17, ef=ef, omp=4)
        run("hnsw", 18, ef=ef, pthreads=4)

def run_threads():
    print("[threads] 全策略加速比横向对比（扫线程数）")
    for t in [1, 2, 4, 8]:
        run("threads",  1, omp=t)
        run("threads",  2, pthreads=t)
        run("threads",  5, rerank_k=500, omp=t)
        run("threads",  6, rerank_k=500, pthreads=t)
        run("threads",  8, nprobe=16, omp=t)
        run("threads",  9, nprobe=16, pthreads=t)
        run("threads", 11, nprobe=16, rerank_k=500, omp=t)
        run("threads", 12, nprobe=16, rerank_k=500, pthreads=t)
        run("threads", 14, nprobe=16, rerank_k=500, omp=t)
        run("threads", 15, nprobe=16, rerank_k=500, pthreads=t)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
GROUPS = {
    "flat":    run_flat,
    "pq":      run_pq,
    "ivf":     run_ivf,
    "ivfpq":   run_ivfpq,
    "hnsw":    run_hnsw,
    "threads": run_threads,
}

requested = sys.argv[1:] if len(sys.argv) > 1 else ["all"]

for g in requested:
    if g == "all":
        for fn in GROUPS.values():
            fn()
    elif g in GROUPS:
        GROUPS[g]()
    else:
        print(f"Unknown group: '{g}'  ({' | '.join(GROUPS)})")
        sys.exit(1)

_csv_file.close()
print(f"\n===== 完成 =====")
print(f"汇总 CSV : {CSV_PATH}")