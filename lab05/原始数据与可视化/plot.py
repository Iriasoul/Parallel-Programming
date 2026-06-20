# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体
# plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
# plt.rcParams["axes.unicode_minus"] = False

CSV = "result.csv"
DPI = 150


def load():
    if not os.path.exists(CSV):
        sys.exit(f"找不到 {CSV}，请先运行 run_bench.py 生成。")
    df = pd.read_csv(CSV)
    # 每查询 CPU 延迟 (us/query)
    df["cpu_us_per_query"] = df["cpu_ms"] / df["M"] * 1000.0
    return df


def pick(df, **kw):
    """取满足条件的唯一一行 (Series)，找不到则告警返回 None。"""
    sub = df
    for k, v in kw.items():
        sub = sub[sub[k] == v]
    if len(sub) == 0:
        print(f"  [warn] 缺少配置 {kw}，该项跳过")
        return None
    return sub.iloc[0]


# 图1: 加速比柱状
def chart_speedup(df):
    items = [
        ("Flat GPU+CPU",        pick(df, strategy=0, M=2000)),
        ("Flat GPU+GPU",        pick(df, strategy=1, M=2000)),
        ("IVF nlist=32",        pick(df, strategy=2, nlist=32,  group_name="by-probeset", group_B=1024)),
        ("IVF nlist=64",        pick(df, strategy=2, nlist=64,  group_name="by-probeset", group_B=1024)),
        ("IVF nlist=256",       pick(df, strategy=2, nlist=256, group_name="sequential",  group_B=1024)),
        ("IVF-PQ global",       pick(df, strategy=3, PQ_M=16, rerank_k=200)),
        ("IVF-PQ per-cluster",  pick(df, strategy=4, PQ_M=16, rerank_k=200)),
    ]
    items = [(n, r) for n, r in items if r is not None]
    labels = [n for n, _ in items]
    vals = [float(r["speedup"]) for _, r in items]
    colors = ["#2e8b57" if v >= 1 else "#c0392b" for v in vals]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = range(len(labels))
    ax.barh(list(y), vals, color=colors, height=0.6)
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.text(1.02, -0.6, "1× (CPU baseline)", color="gray", fontsize=8)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f"{v:.2f}×", va="center", fontsize=9)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Speedup vs corresponding CPU implementation")
    ax.set_title("Speedup of representative configurations")
    ax.set_xlim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig("chart_speedup.png", dpi=DPI)
    plt.close(fig)
    print("  saved chart_speedup.png")


# 图2: 暴力随 M
def chart_flat(df):
    g0 = df[df["strategy"] == 0].sort_values("M")
    g1 = df[df["strategy"] == 1].sort_values("M")
    if len(g0) == 0 or len(g1) == 0:
        print("  [warn] 暴力数据不全，跳过 chart_flat"); return

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(g0["M"], g0["gpu_us_per_query"], "o-", label="GPU+CPU")
    ax.plot(g1["M"], g1["gpu_us_per_query"], "s-", label="GPU+GPU")
    ax.plot(g1["M"], g1["cpu_us_per_query"], "^--", color="gray", label="CPU brute-force")
    ax.set_xscale("log", base=2)
    ax.set_xticks(g1["M"])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Batch size M (queries)")
    ax.set_ylabel("Latency (μs / query)")
    ax.set_title("Brute-force: latency vs batch size")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig("chart_flat.png", dpi=DPI)
    plt.close(fig)
    print("  saved chart_flat.png")


# 图3: IVF 随 nlist
def chart_ivf_nlist(df):
    sub = df[(df["strategy"] == 2) & (df["group_name"] == "sequential") &
             (df["group_B"] == 1024)].sort_values("nlist")
    if len(sub) == 0:
        print("  [warn] IVF nlist 数据不全，跳过 chart_ivf_nlist"); return

    pos = range(len(sub))
    nlist = sub["nlist"].tolist()
    gpu = sub["gpu_us_per_query"].tolist()
    cpu = sub["cpu_us_per_query"].tolist()
    rec = sub["recall"].tolist()
    ov = sub["overlap"].tolist()
    sp = sub["speedup"].tolist()

    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    l1, = ax1.plot(list(pos), gpu, "o-", color="#1f77b4", label="GPU latency")
    l2, = ax1.plot(list(pos), cpu, "s--", color="#ff7f0e", label="CPU latency")
    ax1.set_yscale("log")
    ax1.set_ylabel("Latency (μs / query, log)")
    ax1.set_xlabel("nlist")
    ax1.set_xticks(list(pos))
    ax1.set_xticklabels(nlist)
    # 在 GPU 点旁标注加速比
    for p, gy, s in zip(pos, gpu, sp):
        ax1.annotate(f"{s:.1f}×", (p, gy), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color="#1f77b4")

    ax2 = ax1.twinx()
    l3, = ax2.plot(list(pos), rec, "^-", color="#2ca02c", label="recall")
    l4, = ax2.plot(list(pos), ov, "v-", color="#9467bd", label="overlap")
    ax2.set_ylabel("recall / overlap")
    ax2.set_ylim(0, 1.05)

    ax1.set_title("IVF: latency / recall / overlap / speedup vs nlist")
    ax1.legend(handles=[l1, l2, l3, l4], loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig("chart_ivf_nlist.png", dpi=DPI)
    plt.close(fig)
    print("  saved chart_ivf_nlist.png")


# 图4: 全局 vs 簇内 PQ
def chart_pq(df):
    rr = [50, 100, 200]
    pqm_list = [4, 8, 16]
    cmap = {4: "#1f77b4", 8: "#2ca02c", 16: "#d62728"}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ok = False
    for pqm in pqm_list:
        g = df[(df["strategy"] == 3) & (df["PQ_M"] == pqm)].sort_values("rerank_k")
        p = df[(df["strategy"] == 4) & (df["PQ_M"] == pqm)].sort_values("rerank_k")
        if len(g):
            ax.plot(g["rerank_k"], g["recall"], "o--", color=cmap[pqm],
                    label=f"global PQ_M={pqm}"); ok = True
        if len(p):
            ax.plot(p["rerank_k"], p["recall"], "o-", color=cmap[pqm],
                    label=f"per-cluster PQ_M={pqm}"); ok = True
    if not ok:
        print("  [warn] PQ 数据不全，跳过 chart_pq"); plt.close(fig); return

    # 精确 IVF 参照线 (cpu_recall 即精确召回)
    ref = df[df["strategy"] == 3]["cpu_recall"]
    if len(ref):
        ax.axhline(float(ref.iloc[0]), color="gray", ls=":", lw=1)
        ax.text(rr[0], float(ref.iloc[0]) + 0.005, "exact IVF", color="gray", fontsize=8)

    ax.set_xticks(rr)
    ax.set_xlabel("rerank_k")
    ax.set_ylabel("recall@10")
    ax.set_title("IVF-PQ recall: global (dashed) vs per-cluster (solid)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig("chart_pq.png", dpi=DPI)
    plt.close(fig)
    print("  saved chart_pq.png")


# 分组策略对比
def chart_group(df):
    sub = df[(df["strategy"] == 2) & (df["nlist"] == 64)]
    if len(sub) == 0:
        print("  [warn] 分组数据不全，跳过 chart_group"); return
    modes = ["sequential", "by-primary", "by-probeset", "query-kmeans"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for m in modes:
        s = sub[sub["group_name"] == m].sort_values("group_B")
        if len(s):
            ax.plot(s["group_B"], s["gpu_us_per_query"], "o-", label=m)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(sub["group_B"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("group_B (batch size)")
    ax.set_ylabel("Latency (μs / query)")
    ax.set_title("IVF grouping strategies (nlist=64)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig("chart_group.png", dpi=DPI)
    plt.close(fig)
    print("  saved chart_group.png")


def main():
    import matplotlib.ticker  # noqa: F401  (供 ScalarFormatter 使用)
    df = load()
    print(f"读取 {CSV}: {len(df)} 行")
    chart_speedup(df)
    chart_flat(df)
    chart_ivf_nlist(df)
    chart_pq(df)
    chart_group(df) 
    print("完成。")


if __name__ == "__main__":
    import matplotlib.ticker
    main()
