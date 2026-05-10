import matplotlib.pyplot as plt
import numpy as np

strat4_fix_k500 = {
    "PQ_M": [4, 8, 16, 32],
    "recall": [0.796402, 0.944404, 0.994351, 0.99825],
    "latency": [998.589, 1560.89, 2967.29, 5181.36]
}
strat4_fix_M8 = {
    "rerank_k": [250, 500, 750, 1000],
    "recall": [0.872104, 0.944404, 0.969203, 0.980252],
    "latency": [1325.93, 1560.89, 1799.35, 2090.26]
}
strat5_fix_k250 = {
    "PQ_M": [4, 8, 16],
    "recall": [0.964603, 0.994851, 0.9998],
    "latency": [1214.9, 1295.68, 1670.62]
}
strat5_fix_M4 = {
    "rerank_k": [125, 250, 500],
    "recall": [0.901604, 0.964603, 0.991201],
    "latency": [1023.16, 1214.9, 1409.66]
}

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams["lines.markersize"] = 5
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8

fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=150)
fig.suptitle("参数权衡", fontsize=10, fontweight="bold", y=0.98)

ax1 = axes[0, 0]
x1 = strat4_fix_k500["PQ_M"]
r1 = strat4_fix_k500["recall"]
l1 = strat4_fix_k500["latency"]
ax1_twin = ax1.twinx()
ax1.plot(x1, r1, color="#2E86AB", marker="o")
ax1_twin.plot(x1, l1, color="#F24236", marker="s")
ax1.set_title("s4 | k=500", fontsize=8)
ax1.set_xlabel("M", fontsize=7)
ax1.set_ylabel("召回率", color="#2E86AB", fontsize=7)
ax1_twin.set_ylabel("延迟(us)", color="#F24236", fontsize=7)
ax1.grid(True, linestyle="--", alpha=0.3)

ax2 = axes[0, 1]
x2 = strat4_fix_M8["rerank_k"]
r2 = strat4_fix_M8["recall"]
l2 = strat4_fix_M8["latency"]
ax2_twin = ax2.twinx()
ax2.plot(x2, r2, color="#2E86AB", marker="o")
ax2_twin.plot(x2, l2, color="#F24236", marker="s")
ax2.set_title("s4 | M=8", fontsize=8)
ax2.set_xlabel("k", fontsize=7)
ax2.set_ylabel("召回率", color="#2E86AB", fontsize=7)
ax2_twin.set_ylabel("延迟(us)", color="#F24236", fontsize=7)
ax2.grid(True, linestyle="--", alpha=0.3)

ax3 = axes[1, 0]
x3 = strat5_fix_k250["PQ_M"]
r3 = strat5_fix_k250["recall"]
l3 = strat5_fix_k250["latency"]
ax3_twin = ax3.twinx()
ax3.plot(x3, r3, color="#A23B72", marker="o")
ax3_twin.plot(x3, l3, color="#F18F01", marker="s")
ax3.set_title("s5 | k=250", fontsize=8)
ax3.set_xlabel("M", fontsize=7)
ax3.set_ylabel("召回率", color="#A23B72", fontsize=7)
ax3_twin.set_ylabel("延迟(us)", color="#F18F01", fontsize=7)
ax3.grid(True, linestyle="--", alpha=0.3)

ax4 = axes[1, 1]
x4 = strat5_fix_M4["rerank_k"]
r4 = strat5_fix_M4["recall"]
l4 = strat5_fix_M4["latency"]
ax4_twin = ax4.twinx()
ax4.plot(x4, r4, color="#A23B72", marker="o")
ax4_twin.plot(x4, l4, color="#F18F01", marker="s")
ax4.set_title("s5 | M=4", fontsize=8)
ax4.set_xlabel("k", fontsize=7)
ax4.set_ylabel("召回率", color="#A23B72", fontsize=7)
ax4_twin.set_ylabel("延迟(us)", color="#F18F01", fontsize=7)
ax4.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("v3.png", dpi=150, bbox_inches="tight")
plt.show()