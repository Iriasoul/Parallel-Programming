import matplotlib.pyplot as plt

BASE_LATENCY = 19024.6  
strategy_latency = {
    "Flat": 19024.6,   # 基线
    "Flat-SIMD": 5404.35,
    "SQ-SIDM": 2795.77,
    "PQ-Rerank-SIMD\n（M=8,k=500）": 1560.89, 
    "PQ-Final\n（M=4,k=250）": 1214.9 
}

# 计算加速比
speed_up_ratio = {}
for name, latency in strategy_latency.items():
    speed_up_ratio[name] = BASE_LATENCY / latency

strategy_names = list(speed_up_ratio.keys())
speed_values = list(speed_up_ratio.values())

plt.figure(figsize=(10, 6), dpi=120)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BAR_COLOR = "#1f77b4"
bars = plt.bar(
    strategy_names,
    speed_values,
    color=BAR_COLOR, 
    alpha=0.9,
    linewidth=1.2,
    width=0.6
)

for bar, value in zip(bars, speed_values):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.15,
        f"{value:.2f}",
        ha="center", va="bottom",
        fontsize=12, fontweight="bold"
    )

plt.title("ANN-加速比对比", fontsize=14, pad=20, fontweight="bold")
plt.ylabel("加速比", fontsize=13)
plt.grid(axis='y', linestyle="--", alpha=0.4)
plt.ylim(0, max(speed_values) + 2) 

plt.tight_layout()

plt.savefig("v2.png", dpi=300, bbox_inches="tight")
plt.show()