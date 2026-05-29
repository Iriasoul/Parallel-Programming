"""
Generate 4 charts for Lab3 report.
Run: python3 make_charts.py
Output: chart_speedup.png, chart_thread_scaling.png,
        chart_nprobe_scatter.png, chart_ef_scatter.png
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

# CJK 字体
cjk_candidates = [
    "Noto Sans CJK SC", "Noto Serif CJK SC",
    "PingFang SC", "Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei",
]
available = {f.name for f in font_manager.fontManager.ttflist}
for f in cjk_candidates:
    if f in available:
        mpl.rcParams["font.family"] = f
        break
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130
mpl.rcParams["savefig.dpi"] = 200
mpl.rcParams["savefig.bbox"] = "tight"

S0 = 5761.17

# C1: 各算法最优策略加速比柱状图
def chart_speedup():
    # 选取每个算法的最优 OMP 策略 (vs S0)
    items = [
        ("S0\nFlat (基线)",              5761.17, "#9E9E9E"),
        ("S1\nFlat OMP T=8",              967.10, "#42A5F5"),
        ("S5\nPQ OMP T=8",                450.19, "#42A5F5"),
        ("S8\nIVF OMP T=8",               108.88, "#66BB6A"),
        ("S11\nIVF-PQ M1 OMP T=8",        227.03, "#FFA726"),
        ("S14\nIVF-PQ M2 OMP T=8",        277.75, "#FFA726"),
        ("S17\nHNSW OMP T=4",             288.18, "#AB47BC"),
    ]
    labels = [x[0] for x in items]
    speedups = [S0 / x[1] for x in items]
    colors = [x[2] for x in items]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars = ax.bar(labels, speedups, color=colors, edgecolor="white", width=0.65)

    for bar, sp in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
                f"{sp:.2f}×", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    ax.set_ylabel("加速比 (相对 S0)", fontsize=11)
    ax.set_title("各算法最优加速比", fontsize=12, pad=10)
    ax.set_ylim(0, max(speedups) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    plt.savefig("chart_speedup.png")
    plt.close()
    print("[OK] chart_speedup.png")

# C2: 多线程扩展性 
def chart_thread_scaling():
    threads = [1, 2, 4, 8]
    data = {
        "Flat OMP (S1)":        [5939.98, 2961.62, 1587.22,  967.10],
        "Flat Pthread (S2)":    [6060.98, 3063.28, 1987.00, 1561.44],
        "PQ OMP (S5)":          [1491.61,  857.94,  585.11,  450.19],
        "PQ Pthread (S6)":      [1462.32,  917.87,  667.63,  738.04],
        "IVF OMP (S8)":         [ 312.89,  180.72,  136.95,  108.88],
        "IVF Pthread (S9)":     [ 357.02,  219.71,  176.61,  181.02],
        "IVF-PQ M1 OMP (S11)":  [ 437.14,  326.23,  259.78,  227.03],
        "IVF-PQ M1 Pthread (S12)":[ 446.20,  355.40,  315.29,  316.42],
        "IVF-PQ M2 OMP (S14)":  [ 736.14,  521.18,  380.13,  277.75],
        "IVF-PQ M2 Pthread (S15)":[ 815.68,  615.96,  416.22,  361.83],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    # 左图：OMP（5 条）
    omp_keys = [k for k in data if "OMP" in k]
    pth_keys = [k for k in data if "Pthread" in k]
    palette_omp = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#C62828"]
    palette_pth = ["#64B5F6", "#81C784", "#FFB74D", "#BA68C8", "#E57373"]

    for k, c in zip(omp_keys, palette_omp):
        axes[0].plot(threads, data[k], marker="o", markersize=6,
                     linewidth=1.8, color=c, label=k)
    axes[0].set_title("OpenMP 版本", fontsize=11)
    axes[0].set_xlabel("线程数 T")
    axes[0].set_ylabel("延迟 (µs)")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xticks(threads, [str(t) for t in threads])
    axes[0].legend(fontsize=8, loc="upper right", frameon=False)
    axes[0].grid(True, which="both", linestyle="--", alpha=0.35)
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    for k, c in zip(pth_keys, palette_pth):
        axes[1].plot(threads, data[k], marker="s", markersize=6,
                     linewidth=1.8, color=c, label=k)
    axes[1].set_title("Pthread 版本", fontsize=11)
    axes[1].set_xlabel("线程数 T")
    axes[1].set_ylabel("延迟 (µs)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xticks(threads, [str(t) for t in threads])
    axes[1].legend(fontsize=8, loc="upper right", frameon=False)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.35)
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

    fig.suptitle("多线程扩展性",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig("chart_thread_scaling.png")
    plt.close()
    print("[OK] chart_thread_scaling.png")

# C3: IVF / IVF-PQ nprobe recall-latency 散点
def chart_nprobe_scatter():
    nprobe = [4, 8, 16, 32, 64]
    # (recall, latency) at each nprobe, single thread
    s7  = [(0.8002,   92.22), (0.8986,  174.50), (0.9583,  336.28),
           (0.9876,  645.48), (0.9974, 1272.84)]
    s10 = [(0.7836,  255.67), (0.8654,  348.83), (0.9107,  454.86),
           (0.9311,  631.70), (0.9359,  840.20)]
    s13 = [(0.7945,  347.17), (0.8847,  495.80), (0.9376,  781.17),
           (0.9601, 1197.79), (0.9668, 1893.51)]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for data, label, color, marker in [
        (s7,  "S7  IVF (单线程)",       "#1976D2", "o"),
        (s10, "S10 IVF-PQ M1 (单线程)", "#388E3C", "s"),
        (s13, "S13 IVF-PQ M2 (单线程)", "#C62828", "^"),
    ]:
        xs = [x[1] for x in data]
        ys = [x[0] for x in data]
        ax.plot(xs, ys, color=color, linewidth=1.6, alpha=0.7)
        ax.scatter(xs, ys, s=60, color=color, marker=marker,
                   edgecolor="white", linewidth=1.0, label=label, zorder=5)
        for (rc, lat), np_ in zip(data, nprobe):
            ax.annotate(f"np={np_}", (lat, rc), textcoords="offset points",
                        xytext=(8, -4), fontsize=8, color=color)

    ax.set_xlabel("平均延迟 (µs)")
    ax.set_ylabel("召回率 (Recall@10)")
    ax.set_title("IVF / IVF-PQ - nprobe", fontsize=12)
    ax.set_ylim(0.75, 1.005)
    ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.92)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("chart_nprobe_scatter.png")
    plt.close()
    print("[OK] chart_nprobe_scatter.png")

# C4: HNSW ef recall-latency
def chart_ef_scatter():
    ef_list = [16, 32, 64, 128, 200]
    # (recall, latency)
    s16 = [(0.8708, 122.06), (0.9459, 193.91), (0.9808, 315.70),
           (0.9952, 519.43), (0.9979, 767.35)]
    # S17/S18 ef 固定为 64
    s17_point = (0.9963, 288.18)  # T=4 OMP ef=64
    s18_point = (0.9963, 336.50)  # T=4 Pthread ef=64

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    # S16 曲线
    xs = [x[1] for x in s16]; ys = [x[0] for x in s16]
    ax.plot(xs, ys, color="#1976D2", linewidth=1.8, alpha=0.7)
    ax.scatter(xs, ys, s=70, color="#1976D2", marker="o",
               edgecolor="white", linewidth=1.0,
               label="S16 HNSW 单线程", zorder=5)
    for (rc, lat), ef in zip(s16, ef_list):
        ax.annotate(f"ef={ef}", (lat, rc), textcoords="offset points",
                    xytext=(7, -10), fontsize=8, color="#1976D2")

    # S17/S18 单点
    ax.scatter([s17_point[1]], [s17_point[0]], s=140,
               marker="*", color="#388E3C", edgecolor="white",
               linewidth=1.0, label="S17 HNSW OMP T=4 (ef子图=64)", zorder=6)
    ax.annotate("S17", s17_point[::-1], textcoords="offset points",
                xytext=(10, -3), fontsize=10, color="#388E3C", fontweight="bold")

    ax.scatter([s18_point[1]], [s18_point[0]], s=140,
               marker="*", color="#C62828", edgecolor="white",
               linewidth=1.0, label="S18 HNSW Pthread T=4 (ef子图=64)", zorder=6)
    ax.annotate("S18", s18_point[::-1], textcoords="offset points",
                xytext=(10, -3), fontsize=10, color="#C62828", fontweight="bold")

    # 标注比较参考线
    ax.axhline(0.995, color="gray", linestyle=":", alpha=0.5)
    ax.text(800, 0.997, "99.5% 召回率参考线", fontsize=8, color="gray")

    ax.set_xlabel("平均延迟 (µs)")
    ax.set_ylabel("召回率 (Recall@10)")
    ax.set_title("HNSW - ef_search", fontsize=12)
    ax.set_ylim(0.86, 1.005)
    ax.legend(loc="lower right", fontsize=9.5, frameon=True, framealpha=0.92)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("chart_ef_scatter.png")
    plt.close()
    print("[OK] chart_ef_scatter.png")

if __name__ == "__main__":
    chart_speedup()
    chart_thread_scaling()
    chart_nprobe_scatter()
    chart_ef_scatter()
    print("All charts saved.")