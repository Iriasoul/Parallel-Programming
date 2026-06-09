#!/usr/bin/env python3
# ===========================================================================
#  run_mpi_batch.py —— lab4 MPI 批量测试脚本 (驱动 r_main)
#
#  · 每个配置用 mpiexec -np P ./r_main --flags ... 跑 REPEAT 次, 取延迟最小的一次
#    作为结果 (抵消服务器共享导致的波动, 与 lab3 多次取最小值口径一致)
#  · 结果汇总写入 CSV
#
#  用法:
#    python3 run_mpi_batch.py                 # 跑全部 suite, repeat=3
#    python3 run_mpi_batch.py --suite scaling,comm
#    python3 run_mpi_batch.py --repeat 5 --np-max 8 --out result.csv
#    python3 run_mpi_batch.py --list          # 列出所有 suite
#    python3 run_mpi_batch.py --dry           # 只打印将要执行的命令
#
#  单机(master)直接跑即可; 在 PBS 作业里跑时会自动用 $PBS_NODEFILE 做 -machinefile.
# ===========================================================================
import subprocess, re, csv, sys, os, argparse, time

MPIEXEC = "mpiexec"        # 如需绝对路径改成 "/usr/local/bin/mpiexec"
EXE     = "./r_main"
MACHINEFILE = os.environ.get("PBS_NODEFILE")   # PBS 作业内自动生效; 单机为 None

STRAT_NAME = {
    0:"Serial-IVF-baseline", 1:"MPI-IVF-block-gather", 2:"MPI-IVF-cyclic-gather",
    3:"MPI-IVF-reduce", 4:"MPI-IVF-omp", 5:"MPI-IVF-batch", 6:"MPI-IVF-overlap",
    7:"MPI-IVFPQ-m1", 8:"MPI-IVFPQ-m1-omp", 9:"MPI-IVFPQ-m2", 10:"MPI-IVFPQ-m2-omp",
    11:"MPI-HNSW", 12:"MPI-HNSW-omp", 13:"MPI-IVF-HNSW", 14:"HNSW-on-HNSW",
}

# CSV 列
FIELDS = ["suite","strategy","name","P","omp","part","nprobe","nlist","rerank_k",
          "pqm","ef","topk","test_number","recall","latency_us","qps","n_runs","best_of"]

# ------------------ 测试矩阵 ------------------
# 每个 run-spec 是一个 dict: 必含 strategy, np; 其余 flag 不写则用 r_main 默认值
def spec(strategy, np, **kw):
    d = {"strategy": strategy, "np": np}
    d.update(kw)
    return d

def build_suites(np_list):
    S = {}

    # 1) 扩展性: 主要 MPI-IVF 策略 随 P 变化
    S["scaling"] = (
        [spec(0, 1)] +                                   # 串行基线
        [spec(s, p) for s in (1, 3, 5, 6) for p in np_list]
    )

    # 2) 归并方式: Gather(1) vs 树形Reduce(3)
    S["merge"] = [spec(s, p) for s in (1, 3) for p in np_list]

    # 3) 通信粒度: 逐查询(1) vs 批量(5) vs 非阻塞overlap(6)
    S["comm"] = [spec(s, p) for s in (1, 5, 6) for p in np_list]

    # 4) 划分方式: block(1) vs cyclic(2)
    S["partition"] = [spec(s, p) for s in (1, 2) for p in np_list]

    # 5) MPI×OpenMP 混合: 受 8 核约束, 取 P*T<=8 的组合 (+ 同等总核数的纯MPI对照)
    hybrid = []
    for (p, t) in [(1,1),(1,2),(1,4),(1,8),(2,2),(2,4),(4,2)]:
        if p in np_list or p == 1:
            hybrid.append(spec(4, p, omp=t))
    # 纯 MPI 对照 (策略1, T=1)
    hybrid += [spec(1, p) for p in np_list]
    S["hybrid"] = hybrid

    # 6) IVF-PQ 两方法 ± OpenMP
    S["pq"] = (
        [spec(s, p) for s in (7, 9) for p in np_list] +          # 纯 MPI
        [spec(s, 2, omp=4) for s in (8, 10)]                     # +OMP (np=2,T=4)
    )

    # 7) 图索引
    S["hnsw"] = (
        [spec(11, p) for p in np_list] +
        [spec(12, 2, omp=4)] +
        [spec(13, p) for p in np_list] +
        [spec(14, p) for p in np_list]
    )

    # 8) nprobe 扫描 (IVF 单线程基线视角: 用 np=1 看纯算法权衡)
    S["nprobe"] = [spec(1, 1, nprobe=x) for x in (4, 8, 16, 32, 64)]
    # IVF-PQ 两方法的 nprobe 扫描
    S["nprobe_pq"] = ([spec(7, 1, nprobe=x) for x in (4,8,16,32,64)] +
                      [spec(9, 1, nprobe=x) for x in (4,8,16,32,64)])

    # 9) HNSW ef_search 扫描 (单线程)
    S["ef"] = [spec(11, 1, ef=x) for x in (16, 32, 64, 128, 200)]

    # 10) local_topk 影响 (每 rank 上报候选数, 召回 vs 通信量)
    S["topk"] = [spec(1, p, topk=x) for p in (4, 8) for x in (10, 20, 50)]

    return S

# ------------------ 执行与解析 ------------------
def make_cmd(s):
    cmd = [MPIEXEC, "-np", str(s["np"])]
    if MACHINEFILE:
        cmd += ["-machinefile", MACHINEFILE]
    cmd += [EXE, "--strategy", str(s["strategy"])]
    for key, flag in [("nprobe","--nprobe"),("nlist","--nlist"),("rerank_k","--rerank_k"),
                      ("pqm","--pqm"),("ef","--ef"),("omp","--omp"),
                      ("part","--part"),("topk","--topk"),("test_number","--test_number")]:
        if key in s:
            cmd += [flag, str(s[key])]
    return cmd

def parse_result(stdout):
    line = None
    for ln in stdout.splitlines():
        if ln.startswith("RESULT"):
            line = ln; break
    if not line:
        return None
    d = {}
    for tok in line.split()[1:]:
        if "=" in tok:
            kk, vv = tok.split("=", 1)
            d[kk] = vv
    return d

def run_once(s, timeout=None):
    try:
        p = subprocess.run(make_cmd(s), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    return parse_result(p.stdout)

# ------------------ 主流程 ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all", help="逗号分隔的 suite 名, 或 all")
    ap.add_argument("--repeat", type=int, default=5, help="每配置重复次数, 取最小延迟")
    ap.add_argument("--np-max", type=int, default=8, help="单机时限制最大进程数")
    ap.add_argument("--out", default="result.csv")
    ap.add_argument("--list", action="store_true", help="列出所有 suite 后退出")
    ap.add_argument("--dry", action="store_true", help="只打印命令不执行")
    ap.add_argument("--timeout", type=int, default=600, help="单次运行超时(秒)")
    args = ap.parse_args()

    np_list = [p for p in (1, 2, 4, 8) if p <= args.np_max]
    SUITES = build_suites(np_list)

    if args.list:
        for name, specs in SUITES.items():
            print(f"{name:12s} {len(specs)} configs")
        return

    chosen = list(SUITES.keys()) if args.suite == "all" else args.suite.split(",")
    chosen = [c for c in chosen if c in SUITES]
    if not chosen:
        print("no valid suite; --list to see options", file=sys.stderr); return

    rows = []
    for name in chosen:
        specs = [s for s in SUITES[name] if s["np"] <= args.np_max]
        print(f"\n===== suite: {name} ({len(specs)} configs) =====", file=sys.stderr)
        for s in specs:
            if args.dry:
                print(" ".join(make_cmd(s)), file=sys.stderr); continue
            best = None
            for r in range(args.repeat):
                d = run_once(s, timeout=args.timeout)
                if d is None:
                    continue
                lat = float(d.get("latency_us", "inf"))
                if best is None or lat < float(best.get("latency_us", "inf")):
                    best = d
            if best is None:
                print(f"  [FAIL] strategy={s['strategy']} np={s['np']} {s}", file=sys.stderr)
                continue
            row = {
                "suite": name,
                "strategy": int(best.get("strategy", s["strategy"])),
                "name": STRAT_NAME.get(int(best.get("strategy", s["strategy"])), "?"),
                "P": int(best.get("P", s["np"])),
                "omp": int(best.get("omp", 0)),
                "part": best.get("part", ""),
                "nprobe": int(best.get("nprobe", 0)),
                "nlist": int(best.get("nlist", 0)),
                "rerank_k": int(best.get("rerank_k", 0)),
                "pqm": int(best.get("pqm", 0)),
                "ef": int(best.get("ef", 0)),
                "topk": int(best.get("topk", 0)),
                "test_number": int(best.get("test_number", 0)),
                "recall": round(float(best.get("recall", 0)), 6),
                "latency_us": round(float(best.get("latency_us", 0)), 3),
                "qps": round(float(best.get("qps", 0)), 2),
                "n_runs": args.repeat,
                "best_of": args.repeat,
            }
            rows.append(row)
            print(f"  S{row['strategy']:<2d} {row['name']:<22s} P={row['P']} "
                  f"omp={row['omp']} np_args={ {k:v for k,v in s.items() if k not in ('strategy','np')} } "
                  f"-> recall={row['recall']:.4f} lat={row['latency_us']:.1f}us "
                  f"qps={row['qps']:.0f}", file=sys.stderr)

    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n[done] {len(rows)} rows -> {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
