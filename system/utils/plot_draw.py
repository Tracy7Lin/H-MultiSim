import matplotlib.pyplot as plt
"""结果可视化
        val_acc (list): 主任务准确率序列
        backdoor_acc (list): 后门成功率序列
        epochs (int): 总训练轮次
        training_times (list): 各轮次耗时
        malicious_counts (list): 每轮恶意客户端数量
        save_path (str): 图片存储路径"""
def plot_performance(val_acc, backdoor_acc, epochs, training_times, malicious_counts, save_path='./figs/'):
    # 主任务与后门对比图
    plt.figure()
    plt.rcParams['axes.unicode_minus'] = False
    ax = plt.gca()
    ax.plot(range(len(val_acc)), val_acc, linewidth=1, color='#1f77b4', label='Main Task')
    ax.plot(range(len(backdoor_acc)), backdoor_acc, linewidth=1, linestyle='--', color='#d62728', label='Backdoor')
    ax.set(xlabel='Training Rounds', ylabel='Accuracy (%)', title='Task Performance Comparison')
    ax.legend(loc='best')
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig(f'{save_path}performance_comparison.png', dpi=600)
    plt.close()

    # 训练耗时分析图
    plt.figure()
    plt.rcParams['axes.unicode_minus'] = False
    # 主坐标轴：全局训练耗时（左侧）
    ax1 = plt.gca()
    line1 = ax1.plot(range(len(training_times)), training_times, linewidth=1, color='#1f77b4', label='Time Cost')
    ax1.set(xlabel='Training Rounds', ylabel='Time (sec)', title='Training Analysis')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    # 副坐标轴：恶意客户端数量（右侧）
    ax2 = ax1.twinx()
    line2 = ax2.plot(range(len(malicious_counts)), malicious_counts, linewidth=1, color='#d62728', linestyle='--', label='Malicious Clients')
    ax2.set_ylabel('Number of malicious clients')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig(f'{save_path}training_analysis.png', dpi=600)
    plt.close()
