#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_batch.py —— ANN GPU 批量测试驱动 (原味: 每个配置一个独立进程, 完整冷启动)

流程: 对扫描网格里的每个配置, subprocess 拉起一次 ann(.exe), 它内部完成
      建索引 + 搜索(reps 次取中位数) + CPU 基线, 把一行 CSV 打到 stdout;
      本脚本抓取该行, 增量写入 result.csv。

先编译好二进制:
  nvcc main.cu -o ann -O2 -lcublas -Xcompiler "/openmp /utf-8"
再运行:
  python run_batch.py
"""
import os, sys, subprocess, time

# ---- 二进制位置 (Windows 为 ann.exe) ----
EXE = os.path.abspath("ann.exe" if os.name == "nt" else "./ann")
OUT = "result.csv"
TIMEOUT = 600          # 单个配置超时(秒); 簇内PQ建索引较慢, 留足
DEFAULTS = dict(reps=5, cpu=1)   # 每个配置都带上的公共参数 (可被单配置覆盖)

# =====================================================================
#  扫描网格 (按需增删/注释; 注意簇内PQ(strategy=4)每次重建索引约十几秒)
# =====================================================================
GRID = []

# 1) 暴力检索: 两种 top-k 方案 × batch(M) 扫描  -> 复现 GPU+CPU vs GPU+GPU 交叉
for s in (0, 1):
    for m in (64, 256, 512, 1024, 2000):
        GRID.append(dict(strategy=s, testq=m))

# 2) IVF 精确: nlist × 分组策略 × group_B
for nlist in (32, 64, 128, 256):
    for gm in (0, 1, 2, 3):
        for gB in (256, 512, 1024):
            GRID.append(dict(strategy=2, nlist=nlist, nprobe=16, group_mode=gm, group_B=gB))

# 3) IVF-PQ: 全局(3)/簇内(4) × PQ_M × rerank_k
for s in (3, 4):
    for pqm in (4, 8, 16):
        for rr in (50, 100, 200):
            GRID.append(dict(strategy=s, nlist=64, nprobe=16, group_mode=1,
                             group_B=512, pq_m=pqm, rerank_k=rr))
# =====================================================================


def build_args(cfg):
    args = [EXE]
    merged = dict(DEFAULTS); merged.update(cfg)
    for key, val in merged.items():
        args += ["--" + key, str(val)]
    return args


def get_header():
    r = subprocess.run([EXE, "--header"], capture_output=True, text=True, timeout=30)
    return r.stdout.strip()


def main():
    if not os.path.exists(EXE):
        sys.exit(f"找不到二进制 {EXE}，请先用 nvcc 编译。")

    header = get_header()
    ncol = header.count(",") + 1
    total = len(GRID)
    print(f"二进制: {EXE}\n配置数: {total}\n输出: {OUT}\n")

    t0 = time.time()
    ok = fail = 0
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        f.write(header + "\n"); f.flush()
        for i, cfg in enumerate(GRID, 1):
            tag = " ".join(f"{k}={v}" for k, v in cfg.items())
            print(f"[{i}/{total}] {tag} ...", end=" ", flush=True)
            try:
                r = subprocess.run(build_args(cfg), capture_output=True, text=True, timeout=TIMEOUT)
            except subprocess.TimeoutExpired:
                print("超时, 跳过"); fail += 1; continue
            if r.returncode != 0:
                print(f"失败(rc={r.returncode}), 跳过")
                if r.stderr: print("   stderr:", r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "")
                fail += 1; continue
            row = r.stdout.strip()
            if row.count(",") + 1 != ncol:
                print("输出列数不符, 跳过"); fail += 1; continue
            f.write(row + "\n"); f.flush()      # 增量写入, 崩溃也不丢已完成的
            # 抽取 recall / gpu_ms / speedup 简报 (列序见 header)
            c = row.split(","); 
            print(f"recall={c[13]} gpu={c[15]}ms speedup={c[17]}x")
            ok += 1

    dt = time.time() - t0
    print(f"\n完成: 成功 {ok}, 失败 {fail}, 用时 {dt:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
