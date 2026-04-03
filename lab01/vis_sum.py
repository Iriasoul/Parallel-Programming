import matplotlib.pyplot as plt
import math

def load_data(file_path):
    N_list = []
    plain_list = []
    two_list = []
    four_list = []

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
        elif line.startswith('两路优化'):
            time = float(line.split(':')[-1].replace('us', '').strip())
            two_list.append(time)
        elif line.startswith('4路循环展开'):
            time = float(line.split(':')[-1].replace('us', '').strip())
            four_list.append(time)

    return N_list, plain_list, two_list, four_list

def plot_performance(Ns, plain, two, four):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 7))

    exponents = [int(math.log2(n)) for n in Ns]
    
    plt.plot(exponents, plain, 'o-', color='#E63946', linewidth=1.25, markersize=8, label='平凡算法')
    plt.plot(exponents, two, 's-', color='#0052CC', linewidth=1.25, markersize=8, label='两路优化')
    plt.plot(exponents, four, '^-', color='#FF7D00', linewidth=1.25, markersize=8, label='4路循环展开')

    plt.xticks(exponents, [f'$2^{{{k}}}$' for k in exponents])
    plt.yscale('log')
    plt.xlabel('数据规模 N', fontsize=14, fontweight='bold')
    plt.ylabel('运行时间 (us)', fontsize=14, fontweight='bold')
    plt.title('不同数据大小下优化算法性能对比', fontsize=16, fontweight='bold', pad=20)

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.savefig('sum_vis.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    FILE_PATH = 'sum_result.txt'
    Ns, plain, two, four = load_data(FILE_PATH)
    plot_performance(Ns, plain, two, four)