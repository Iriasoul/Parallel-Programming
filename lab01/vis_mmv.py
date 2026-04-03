import matplotlib.pyplot as plt

def load_data(file_path):
    N_list = []
    plain_list = []
    cache_list = []
    loop_list = []

    with open(file_path, 'r', encoding='gbk') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('---'):
            continue
        if line.startswith('N ='):
            n = int(line.split('=')[-1].strip())
            N_list.append(n)
        elif line.startswith('平凡算法'):
            time = float(line.split(':')[-1].replace('us', '').strip())
            plain_list.append(time)
        elif line.startswith('Cache优化'):
            time = float(line.split(':')[-1].replace('us', '').strip())
            cache_list.append(time)
        elif line.startswith('循环展开'):
            time = float(line.split(':')[-1].replace('us', '').strip())
            loop_list.append(time)

    return N_list, plain_list, cache_list, loop_list

def plot_performance(Ns, plain, cache, loop):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 7))

    plt.plot(Ns, plain, 'o-', color="#169B0F", linewidth=1.25, markersize=8, label='平凡算法')
    plt.plot(Ns, cache, 's-', color='#0052CC', linewidth=1.25, markersize=8, label='Cache优化')
    plt.plot(Ns, loop, '^-', color='#FF7D00', linewidth=1.25, markersize=8, label='循环展开')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('矩阵大小 N', fontsize=14, fontweight='bold')
    plt.ylabel('运行时间 (us)', fontsize=14, fontweight='bold')
    plt.title('不同矩阵大小下算法性能对比', fontsize=16, fontweight='bold', pad=20)

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.savefig('mmv_vis.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    FILE_PATH = 'mmv_result.txt'
    Ns, plain, cache, loop = load_data(FILE_PATH)
    plot_performance(Ns, plain, cache, loop)