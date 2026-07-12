# plot.py —— 读 results/all.csv 数据可视化
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "results")
rows = list(csv.DictReader(open(f"{OUT}/all.csv", encoding="utf-8")))
def F(x):
    try: return float(x)
    except: return None

# 每方法一种颜色/marker
STYLE = {
    "CPU-IVF":        ("#1f77b4", "o"),
    "CPU-IVF-PQ":     ("#17becf", "v"),
    "LSH-CPU":        ("#ff7f0e", "s"),
    "LSH-GPU":        ("#d62728", "D"),
    "GPU-flat":       ("#2ca02c", "*"),
    "hetero-exact":   ("#bcbd22", "h"),   # 融合一 exact 对照
    "hetero-LSH":     ("#9467bd", "P"),   # 融合一 GPU-LSH
    "MPI-exact":      ("#17becf", "X"),   # 融合二 exact 对照
    "MPI-LSH":        ("#8c564b", "X"),   # 融合二 GPU-LSH
}

# ---------- 图1：Latency-Recall@10 Pareto ----------
fig, ax = plt.subplots(figsize=(9, 6))
pts = []  # (us, recall) for frontier
for meth, (c, mk) in STYLE.items():
    xs = [F(r["us_per_query"]) for r in rows if r["method"] == meth]
    ys = [F(r["recall"])       for r in rows if r["method"] == meth]
    xs, ys = zip(*[(x, y) for x, y in zip(xs, ys) if x and y]) if any(xs) else ([], [])
    if not xs: continue
    ax.scatter(xs, ys, c=c, marker=mk, s=70, label=meth, edgecolors="k", linewidths=0.4, zorder=3)
    pts += list(zip(xs, ys))

# Pareto 前沿（低延迟高召回更优）：按延迟升序，保留 recall 递增的点
pts.sort()
front, best = [], -1
for x, y in pts:
    if y > best: front.append((x, y)); best = y
if front:
    fx, fy = zip(*front)
    ax.plot(fx, fy, "k--", lw=1.2, alpha=0.6, zorder=2, label="Pareto frontier")

ax.set_xscale("log")
ax.set_xlabel("Latency (us / query, log scale)")
ax.set_ylabel("Recall@10")
ax.set_title("ANNS on DEEP100K (x86 + RTX4060): Latency-Recall@10 Pareto")
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/pareto.png", dpi=140)
print("wrote", f"{OUT}/pareto.png")

# ---------- 图2：各方法最优加速比 ----------
best_sp = {}
for r in rows:
    sp = F(r["speedup"]); rec = F(r["recall"])
    if sp is None or rec is None: continue
    if rec < 0.95: continue
    best_sp[r["method"]] = max(best_sp.get(r["method"], 0), sp)
order = sorted(best_sp, key=best_sp.get)
fig2, ax2 = plt.subplots(figsize=(9, 5))
colors = [STYLE.get(m, ("#333", "o"))[0] for m in order]
bars = ax2.barh(order, [best_sp[m] for m in order], color=colors, edgecolor="k")
for b, m in zip(bars, order):
    ax2.text(b.get_width()+2, b.get_y()+b.get_height()/2, f"{best_sp[m]:.0f}x", va="center", fontsize=9)
ax2.set_xlabel("Best speedup vs brute-force AVX2 baseline (1562.5 us/q)")
ax2.set_title("Best speedup per method (DEEP100K, x86 + RTX4060)")
ax2.grid(True, axis="x", ls=":", alpha=0.5)
fig2.tight_layout(); fig2.savefig(f"{OUT}/speedup.png", dpi=140)
print("wrote", f"{OUT}/speedup.png")

# ---------- 图3：LSH-GPU 参数扫描 (同图1坐标：x=Latency log, y=Recall) ----------
lsh_rows = [r for r in rows if r["method"] == "LSH-GPU"]
parsed = []
for r in lsh_rows:
    label = r["label"]
    parts = label.split(",")
    k = None; rk = None
    for p in parts:
        p = p.strip()
        if p.startswith("K="): k = int(p[2:])
        if p.startswith("rk="): rk = int(p[3:])
    rec = F(r["recall"]); lat = F(r["us_per_query"])
    if k is not None and rk is not None and rec and lat:
        parsed.append((k, rk, rec, lat))

# 按 K 分颜色，按 rk 分标记 —— 与图1风格一致
k_colors = {64: "#d62728", 128: "#ff7f0e", 256: "#1f77b4"}
rk_markers = {200: "o", 500: "s", 1000: "D"}
rk_sizes  = {200: 70, 500: 90, 1000: 110}

fig3, ax3 = plt.subplots(figsize=(9, 6))
for k, rk, rec, lat in parsed:
    c = k_colors.get(k, "#333")
    mk = rk_markers.get(rk, "o")
    sz = rk_sizes.get(rk, 70)
    ax3.scatter(lat, rec, c=c, marker=mk, s=sz, edgecolors="k", linewidths=0.4, zorder=3)

# 用线连接同一 K 的 rk=200→500→1000 便于追踪趋势
for k in sorted(k_colors):
    pts = sorted([p for p in parsed if p[0] == k], key=lambda x: x[1])
    if len(pts) >= 2:
        xs = [p[3] for p in pts]; ys = [p[2] for p in pts]
        ax3.plot(xs, ys, "k-", lw=0.8, alpha=0.3, zorder=1)

# 标注每个点
for k, rk, rec, lat in parsed:
    ax3.annotate(f"(K={k},rk={rk})", (lat, rec), textcoords="offset points",
                 xytext=(5, 5), fontsize=6.5, alpha=0.8)

ax3.set_xlabel("Latency (us / query)")
ax3.set_ylabel("Recall@10")
ax3.set_title("LSH-GPU Parameter Sweep (DEEP100K, RTX4060)")
ax3.grid(True, which="both", ls=":", alpha=0.5)

from matplotlib.lines import Line2D
leg_k = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, markeredgecolor="k", markeredgewidth=0.4, label=f"K={k}") for k, c in sorted(k_colors.items())]
leg_rk = [Line2D([0], [0], marker=mk, color="w", markerfacecolor="#555", markersize=8, markeredgecolor="k", markeredgewidth=0.4, label=f"rk={rk}") for rk, mk in sorted(rk_markers.items())]
leg1 = ax3.legend(handles=leg_k, title="K (hash bits)", fontsize=8, loc="lower left")
ax3.add_artist(leg1)
ax3.legend(handles=leg_rk, title="rk (rerank)", fontsize=8, loc="lower right")

fig3.tight_layout(); fig3.savefig(f"{OUT}/lsh_gpu_sweep.png", dpi=140)
print("wrote", f"{OUT}/lsh_gpu_sweep.png")

# ---------- 图4/5/6：异构分流 GPU 占比 f 分析（三张独立图） ----------
def parse_frac(label):
    for part in label.split(","):
        part = part.strip()
        if part.startswith("frac="):   return float(part[5:])
        if part.startswith("gpu_frac="): return float(part[9:])
    return None

h_exact = [(parse_frac(r["label"]), F(r["qps"]), F(r["recall"]))
           for r in rows if r["method"] == "hetero-exact" and parse_frac(r["label"]) is not None]
h_lsh   = [(parse_frac(r["label"]), F(r["qps"]), F(r["recall"]))
           for r in rows if r["method"] == "hetero-LSH"   and parse_frac(r["label"]) is not None]
mpi_exact = [(parse_frac(r["label"]), F(r["qps"]), F(r["recall"]))
             for r in rows if r["method"] == "MPI-exact"   and parse_frac(r["label"]) is not None]
mpi_lsh   = [(parse_frac(r["label"]), F(r["qps"]), F(r["recall"]))
             for r in rows if r["method"] == "MPI-LSH"     and parse_frac(r["label"]) is not None]
h_exact.sort(key=lambda x: x[0]); h_lsh.sort(key=lambda x: x[0])
mpi_exact.sort(key=lambda x: x[0]); mpi_lsh.sort(key=lambda x: x[0])

# 基线 QPS 和 Recall
gpu_lsh_qps   = next((q for f, q, _ in h_lsh   if abs(f - 1.0) < 0.01), 107330)
gpu_exact_qps = next((q for f, q, _ in h_exact if abs(f - 1.0) < 0.01), 40148)
cpu_ivf_qps   = next((q for f, q, _ in h_lsh   if abs(f - 0.0) < 0.01), 37034)
rec_gpu_lsh   = next((r for f, _, r in h_lsh   if abs(f - 1.0) < 0.01), 0.98)
rec_cpu_ivf   = next((r for f, _, r in h_lsh   if abs(f - 0.0) < 0.01), 0.96)
best_lsh   = max(h_lsh,   key=lambda x: x[1])
best_exact = max(h_exact, key=lambda x: x[1])
best_mpi   = max(mpi_lsh, key=lambda x: x[1]) if mpi_lsh else None
fs_l, qps_l, recs_l = zip(*h_lsh)
fs_e, qps_e, recs_e = zip(*h_exact)

# ============ 图4：QPS vs f（LSH 与 exact 左右对比） ============
fig4, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.5))

# (a) hetero-LSH
ax_a.plot(fs_l, qps_l, color="#9467bd", marker="D", markersize=7, linewidth=2,
          markeredgecolor="k", markeredgewidth=0.4, zorder=3)
ax_a.axhline(y=gpu_lsh_qps, color="#d62728", ls="--", lw=1.2, alpha=0.7,
             label=f"GPU-only LSH ({gpu_lsh_qps:.0f} QPS)")
ax_a.axhline(y=cpu_ivf_qps, color="#1f77b4", ls="--", lw=1.2, alpha=0.7,
             label=f"CPU-only IVF ({cpu_ivf_qps:.0f} QPS)")
ax_a.axvline(x=best_lsh[0], color="gray", ls=":", lw=1.0, alpha=0.5)
ax_a.annotate(f"$f^*$={best_lsh[0]:.2f}\n{best_lsh[1]:.0f} QPS",
              xy=(best_lsh[0], best_lsh[1]), xytext=(best_lsh[0] - 0.20, best_lsh[1] - 18000),
              fontsize=10, fontweight="bold", color="#9467bd",
              arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1.5),
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#9467bd", alpha=0.85))
ax_a.axvspan(0.75, 0.85, alpha=0.07, color="#9467bd")
ax_a.text(0.80, gpu_lsh_qps * 0.45, "GPU & CPU finish together",
          ha="center", fontsize=8.5, color="#9467bd", alpha=0.7,
          bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#9467bd", alpha=0.5))
ax_a.set_xlabel("GPU fraction $f$"); ax_a.set_ylabel("QPS")
ax_a.set_title("S5 hetero-LSH  (GPU-LSH  $\parallel$  CPU-IVF)", fontweight="bold")
ax_a.legend(fontsize=8.5, framealpha=0.85)
ax_a.grid(True, ls=":", alpha=0.4)
ax_a.set_xlim(-0.02, 1.08)

# (b) hetero-exact
ax_b.plot(fs_e, qps_e, color="#bcbd22", marker="h", markersize=7, linewidth=2,
          markeredgecolor="k", markeredgewidth=0.4, zorder=3)
ax_b.axhline(y=gpu_exact_qps, color="#d62728", ls="--", lw=1.2, alpha=0.7,
             label=f"GPU-only exact ({gpu_exact_qps:.0f} QPS)")
ax_b.axhline(y=cpu_ivf_qps, color="#1f77b4", ls="--", lw=1.2, alpha=0.7,
             label=f"CPU-only IVF ({cpu_ivf_qps:.0f} QPS)")
ax_b.axvline(x=best_exact[0], color="gray", ls=":", lw=1.0, alpha=0.5)
ax_b.annotate(f"$f^*$={best_exact[0]:.2f}\n{best_exact[1]:.0f} QPS",
              xy=(best_exact[0], best_exact[1]), xytext=(best_exact[0] + 0.15, best_exact[1] + 4000),
              fontsize=10, fontweight="bold", color="#bcbd22",
              arrowprops=dict(arrowstyle="->", color="#bcbd22", lw=1.5),
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bcbd22", alpha=0.85))
ax_b.set_xlabel("GPU fraction $f$"); ax_b.set_ylabel("QPS")
ax_b.set_title("S5 hetero-exact  (GPU-cuBLAS  $\parallel$  CPU-IVF)", fontweight="bold")
ax_b.legend(fontsize=8.5, framealpha=0.85)
ax_b.grid(True, ls=":", alpha=0.4)
ax_b.set_xlim(-0.02, 1.08)

fig4.suptitle("S5 Heterogeneous Offloading: QPS vs GPU fraction $f$  (DEEP100K, x86 + RTX4060)",
              fontsize=12, fontweight="bold", y=1.02)
fig4.tight_layout(); fig4.savefig(f"{OUT}/gpu_frac_qps.png", dpi=150)
print("wrote", f"{OUT}/gpu_frac_qps.png")

# ============ 图5：Recall vs f ============
fig5, ax5 = plt.subplots(figsize=(7, 5))
ax5.plot(fs_l, recs_l, color="#9467bd", marker="D", markersize=7, linewidth=2,
         markeredgecolor="k", markeredgewidth=0.4, label="hetero-LSH")
ax5.plot(fs_e, recs_e, color="#bcbd22", marker="h", markersize=7, linewidth=2,
         markeredgecolor="k", markeredgewidth=0.4, label="hetero-exact")
ax5.axhline(y=rec_gpu_lsh, color="#d62728", ls="--", lw=1.0, alpha=0.5,
            label=f"GPU-only LSH ({rec_gpu_lsh:.4f})")
ax5.axhline(y=rec_cpu_ivf, color="#1f77b4", ls="--", lw=1.0, alpha=0.5,
            label=f"CPU-only IVF ({rec_cpu_ivf:.4f})")
ax5.set_xlabel("GPU fraction $f$"); ax5.set_ylabel("Recall@10")
ax5.set_title("S5 Recall@10 vs GPU fraction $f$", fontweight="bold")
ax5.legend(fontsize=9, framealpha=0.85)
ax5.grid(True, ls=":", alpha=0.4)
ax5.set_xlim(-0.02, 1.08)
fig5.tight_layout(); fig5.savefig(f"{OUT}/gpu_frac_recall.png", dpi=150)
print("wrote", f"{OUT}/gpu_frac_recall.png")

# ============ 图6：S5 vs S6 最优 f 对比 ============
fig6, ax6 = plt.subplots(figsize=(7.5, 5))
ax6.plot(fs_l, qps_l, color="#9467bd", marker="D", markersize=7, linewidth=1.8,
         markeredgecolor="k", markeredgewidth=0.4, label="S5 (std::thread)", zorder=3)
if mpi_lsh:
    fs_m, qps_m, _ = zip(*mpi_lsh)
    ax6.plot(fs_m, qps_m, color="#8c564b", marker="X", markersize=9, linewidth=1.8,
             markeredgecolor="k", markeredgewidth=0.4, label="S6 (MPI, n=2)", zorder=3)

ax6.axvline(x=best_lsh[0], color="#9467bd", ls=":", lw=1.0, alpha=0.4)
ax6.annotate(f"S5  $f^*$={best_lsh[0]:.2f}",
             xy=(best_lsh[0], best_lsh[1] * 0.91),
             fontsize=10, color="#9467bd", fontweight="bold", ha="center")
if best_mpi:
    ax6.axvline(x=best_mpi[0], color="#8c564b", ls=":", lw=1.0, alpha=0.4)
    ax6.annotate(f"S6  $f^*$={best_mpi[0]:.2f}",
                 xy=(best_mpi[0], best_mpi[1] * 0.87),
                 fontsize=10, color="#8c564b", fontweight="bold", ha="center")
    ax6.annotate("S6 $f^*$ < S5 $f^*$\nMPI gives CPU its own process, IVF slightly faster\n$\Rightarrow$ CPU can take more queries",
                 xy=(0.62, best_mpi[1] * 0.42), fontsize=9, color="#555",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.75))

ax6.set_xlabel("GPU fraction $f$"); ax6.set_ylabel("QPS")
ax6.set_title("S5 vs S6: LSH mode optimal $f$ comparison", fontweight="bold")
ax6.legend(fontsize=9, framealpha=0.85)
ax6.grid(True, ls=":", alpha=0.4)
ax6.set_xlim(-0.02, 1.08)
fig6.tight_layout(); fig6.savefig(f"{OUT}/gpu_frac_s5vs6.png", dpi=150)
print("wrote", f"{OUT}/gpu_frac_s5vs6.png")
