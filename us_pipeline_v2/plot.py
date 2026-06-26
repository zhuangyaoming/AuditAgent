import numpy as np
import matplotlib.pyplot as plt
import math

# 全局字体: Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# ==================== 1. 数据准备 ====================
token_ranges = ['13k-26k', '26k-39k', '39k-52k', '52k-65k', '65k-78k']

data = {
    'US': {
        'AuditAgent': {
            'R_I': [0.322, 0.285, 0.348, 0.352, 0.318],
            'R_E': [0.115, 0.102, 0.135, 0.142, 0.102]
        },
        'GPagent': {
            'R_I': [0.131, 0.109, 0.127, 0.122, 0.101],
            'R_E': [0.064, 0.051, 0.053, 0.049, 0.051]
        },
        'SingleLLM': {
            'R_I': [0.225, 0.182, 0.195, 0.141, 0.138],
            'R_E': [0.078, 0.061, 0.071, 0.053, 0.044]
        }
    },
    'CN': {
        'AuditAgent': {
            'R_I': [0.231, 0.202, 0.251, 0.255, 0.228],
            'R_E': [0.16, 0.14, 0.188, 0.195, 0.142]
        },
        'GPagent': {
            'R_I': [0.089, 0.071, 0.083, 0.081, 0.067],
            'R_E': [0.036, 0.024, 0.033, 0.025, 0.027]
        },
        'SingleLLM': {
            'R_I': [0.152, 0.120, 0.131, 0.092, 0.090],
            'R_E': [0.069, 0.054, 0.062, 0.047, 0.039]
        }
    }
}

# 行定义: (metric, market)
row_defs = [
    ('R_I', 'CN'),
    ('R_I', 'US'),
    ('R_E', 'CN'),
    ('R_E', 'US'),
]

# 指标显示名 (LaTeX 下标, \mathrm 保持正体)
metric_labels = {
    'R_I': r'$\mathrm{R}_I$',
    'R_E': r'$\mathrm{R}_E$',
}

models = ['AuditAgent', 'GPagent', 'SingleLLM']
model_labels = {
    'AuditAgent': 'AUDITAGENT',
    'GPagent':    'General-Purpose Agent',
    'SingleLLM':  'Single LLM',
}
model_colors = {
    'AuditAgent': '#aecfe6',
    'GPagent':    '#a8d8a8',
    'SingleLLM':  '#e5989b',
}
model_hatches = {
    'AuditAgent': '',
    'GPagent':    '\\\\',
    'SingleLLM':  'xx',
}

# ==================== 2. 辅助函数 ====================
def get_nice_y_limit(max_val):
    """Y轴上限：取 max*1.15 向上匹配到最近的 nice 值"""
    target = max_val
    for limit in [0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.48, 0.56]:
        if target <= limit:
            return limit
    return math.ceil(target * 10) / 10.0

# ==================== 3. 绘图 ====================
fig, axes = plt.subplots(4, 3, figsize=(18, 14))
plt.subplots_adjust(hspace=0.75, wspace=0.18)

alphabet = [f"({chr(97 + i)})" for i in range(12)]  # (a)–(l)

x = np.arange(len(token_ranges))  # [0, 1, 2, 3, 4]
bar_width = 0.72

plot_idx = 0
for row_idx, (metric, market) in enumerate(row_defs):
    # 该行统一Y轴上限 (同一行内3个模型可公平对比)
    row_all_vals = [data[market][m][metric][c] for m in models for c in range(5)]
    row_y_limit = get_nice_y_limit(max(row_all_vals))

    for col_idx, model in enumerate(models):
        ax = axes[row_idx, col_idx]
        y_vals = data[market][model][metric]

        color = model_colors[model]
        hatch = model_hatches[model]

        # 柱状图
        ax.bar(x, y_vals, bar_width, color=color, edgecolor='#333333',
               linewidth=0.8, hatch=hatch, zorder=3)

        # 顶部折线连接
        ax.plot(x, y_vals, color='#333333', linestyle='--', marker='o',
                linewidth=1.0, markersize=4, zorder=4)

        # X轴 — token ranges
        ax.set_xticks(x)
        ax.set_xticklabels(token_ranges, rotation=20, fontsize=8.5, color='#333333')

        # Y轴 — 统一上限
        ax.set_ylim(0, row_y_limit)
        y_ticks = np.linspace(0, row_y_limit, 5)
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        # 边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')

        # 网格
        ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cccccc', zorder=0)
        ax.tick_params(axis='both', which='major', labelsize=8.5)

        # 子图标题: (a) AUDITAGENT - R_I of FinFraud-CN
        market_label = 'FinFraud-CN' if market == 'CN' else 'FinFraud-US'
        sub_title = f"{alphabet[plot_idx]} {model_labels[model]} - {metric_labels[metric]} of {market_label}"
        ax.set_title(sub_title, fontsize=10.5, fontweight='bold',
                     color='#111111', pad=8)

        plot_idx += 1

plt.savefig('Table4_Visualization_Compact.png', dpi=300, bbox_inches='tight')
plt.show()
