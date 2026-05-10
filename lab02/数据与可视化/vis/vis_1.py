import matplotlib.pyplot as plt

strategy_data = {
    "0 - Flat": [{"latency": 19024.6, "recall": 0.99995, "param": ""}],
    "1 - Flat-SIMD": [{"latency": 5404.35, "recall": 0.99995, "param": ""}],
    "2 - SQ-SIMD": [{"latency": 2795.77, "recall": 0.963406, "param": ""}],
    "3 - PQ-SIMD": [
        {"latency": 554.338, "recall": 0.11475, "param": "M=4"},
        {"latency": 1084.04, "recall": 0.2543, "param": "M=8"},
        {"latency": 2365.27, "recall": 0.433, "param": "M=16"},
        {"latency": 4562.5, "recall": 0.4973, "param": "M=32"},
    ],
    "4 - PQ-Rerank-SIMD": [
        {"latency": 998.589, "recall": 0.796402, "param": "M=4,k=500"},
        {"latency": 1560.89, "recall": 0.944404, "param": "M=8,k=500"},
        {"latency": 2967.29, "recall": 0.994351, "param": "M=16,k=500"},
        {"latency": 1325.93, "recall": 0.872104, "param": "M=8,k=250"},
        {"latency": 2090.26, "recall": 0.980252, "param": "M=8,k=1000"},
    ],
    "5 - PQ-Final": [
        {"latency": 1114.09, "recall": 0.980002, "param": "M=8,k=125"},
        {"latency": 1023.16, "recall": 0.901604, "param": "M=4,k=125"},
        {"latency": 1453.95, "recall": 0.99885, "param": "M=16,k=125"},
        {"latency": 1008.9, "recall": 0.831704, "param": "M=8,k=25"},
    ]
}

strategy_colors = {
    "0 - Flat": "#FF3333",   # 红
    "1 - Flat-SIMD": "#3333FF",   # 蓝
    "2 - SQ-SIMD": "#33CC33",   # 绿
    "3 - PQ-SIMD": "#FF9900",   # 橙
    "4 - PQ-Rerank-SIMD": "#9933CC",   # 紫
    "5 - PQ-Final": "#CC6633"    # 棕
}

plt.figure(figsize=(12, 8), dpi=100)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]  # 中文
plt.rcParams["axes.unicode_minus"] = False 

# 绘制
for strategy, points in strategy_data.items():
    # 排序
    points_sorted = sorted(points, key=lambda p: p["latency"])
    x = [p["latency"] for p in points_sorted]
    y = [p["recall"] for p in points_sorted]
    params = [p["param"] for p in points_sorted]
    color = strategy_colors[strategy]

    # 散点
    plt.scatter(x, y, color=color, s=60, alpha=0.8, label=f"{strategy}")
    # 连线
    plt.plot(x, y, color=color, linewidth=1.5, alpha=0.6)
    # 参数标注
    for xi, yi, param in zip(x, y, params):
        if param:
            plt.annotate(
                param,
                xy=(xi, yi),
                xytext=(8, -7),
                textcoords="offset points",
                fontsize=8,
                color=color
            )

plt.xlabel("平均延迟 (μs)", fontsize=12)
plt.ylabel("平均召回率", fontsize=12)
plt.title("ANN-测试数据", fontsize=14, pad=20)
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(loc="best", fontsize=10) 

plt.tight_layout()
plt.savefig("v1.png", dpi=300)
plt.show()