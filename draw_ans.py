import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
plt.rcParams.update({"font.size": 18})

# 数据
init_labeled = [1000, 2000, 5000, 10000]
increased_labeled = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
# # 原始结果
# RandomSampling = [
#     [0.3558, 0.4026, 0.4364, 0.4674, 0.4062, 0.5172, 0.5522, 0.56, 0.5206, 0.5693, 0.5843],
#     [0.3913, 0.46, 0.4784, 0.4665, 0.529, 0.5383, 0.5284, 0.5603, 0.5638, 0.5823, 0.5895],
#     [0.5023, 0.5356, 0.5373, 0.559, 0.5356, 0.5624, 0.5999, 0.5892, 0.5373, 0.6048, 0.6134],
#     [0.5685, 0.5763, 0.5696, 0.4575, 0.6111, 0.616, 0.6129, 0.6214, 0.6024, 0.6322, 0.6302]
# ]
# LeastConfidence = [
#     [0.3558, 0.2969, 0.4149, 0.4554, 0.475, 0.5162, 0.4549, 0.5528, 0.5673, 0.5494, 0.5742],
#     [0.3913, 0.4433, 0.4911, 0.429, 0.5008, 0.522, 0.5478, 0.5442, 0.4878, 0.5679, 0.5653],
#     [0.5023, 0.5162, 0.4728, 0.5461, 0.5325, 0.5733, 0.5866, 0.5677, 0.5001, 0.5809, 0.5778],
#     [0.5685, 0.5484, 0.5794, 0.5876, 0.5962, 0.6189, 0.6285, 0.6127, 0.5789, 0.639, 0.62]
# ]
# MarginSampling = [
#     [0.3558, 0.3686, 0.4483, 0.4886, 0.446, 0.5098, 0.5272, 0.5508, 0.5541, 0.5282, 0.5849],
#     [0.3913, 0.4453, 0.4678, 0.4841, 0.5193, 0.5321, 0.554, 0.5574, 0.4772, 0.5856, 0.5895],
#     [0.5016, 0.5094, 0.5, 0.5524, 0.5665, 0.5373, 0.5836, 0.5831, 0.4821, 0.6113, 0.5956],
#     [0.5265, 0.6028, 0.5706, 0.5125, 0.612, 0.6097, 0.6122, 0.6006, 0.6059, 0.6349, 0.6359]
# ]
# EntropySampling = [
#     [0.3662, 0.3469, 0.4445, 0.4574, 0.2757, 0.4755, 0.4924, 0.5437, 0.5555, 0.5367, 0.5803],
#     [0.3885, 0.4496, 0.4787, 0.242, 0.5042, 0.5071, 0.5547, 0.5392, 0.514, 0.5544, 0.546],
#     [0.5016, 0.525, 0.5016, 0.5177, 0.549, 0.546, 0.5883, 0.5645, 0.5851, 0.5719, 0.59],
#     [0.5265, 0.5954, 0.5985, 0.5204, 0.6006, 0.6028, 0.6204, 0.6222, 0.5549, 0.6271, 0.6067]
# ]
# full_learning = [
#     [0.3662, 0.3885, 0.4547, 0.4846, 0.5016, 0.5141, 0.4857, 0.5595, 0.5539, 0.5265, 0.5711],
#     [0.3885, 0.4547, 0.4846, 0.5016, 0.5141, 0.4857, 0.5595, 0.5539, 0.5265, 0.5711, 0.5626],
#     [0.5016, 0.5141, 0.4857, 0.5595, 0.5539, 0.5265, 0.5711, 0.5626, 0.4353, 0.606, 0.5983],
#     [0.5265, 0.5711, 0.5626, 0.4353, 0.606, 0.5983, 0.6147, 0.6316, 0.6195, 0.6409, 0.6255]
# ]

# 优化参数结果
RandomSampling = [
    [0.3667, 0.4344, 0.4814, 0.5042, 0.5213, 0.5368, 0.5622, 0.5534, 0.5708, 0.5907, 0.5979],
    [0.4341, 0.4587, 0.5021, 0.5201, 0.5263, 0.5487, 0.5636, 0.5674, 0.5728, 0.5901, 0.6018],
    [0.5152, 0.5428, 0.549, 0.5534, 0.5771, 0.5825, 0.5933, 0.6018, 0.6036, 0.6038, 0.6151],
    [0.593, 0.5946, 0.5958, 0.6092, 0.6205, 0.617, 0.6177, 0.6301, 0.6319, 0.6405, 0.6506]
]
LeastConfidence = [
    [0.3667, 0.4062, 0.4504, 0.4913, 0.5036, 0.545, 0.5563, 0.5653, 0.5727, 0.5769, 0.593],
    [0.4341, 0.466, 0.4775, 0.5105, 0.5374, 0.5524, 0.5677, 0.5843, 0.5843, 0.6049, 0.605],
    [0.5152, 0.5223, 0.5477, 0.5792, 0.5849, 0.5752, 0.5906, 0.6103, 0.6144, 0.6204, 0.6309],
    [0.593, 0.5936, 0.607, 0.5999, 0.6256, 0.6167, 0.6366, 0.6198, 0.6326, 0.6377, 0.6592]
]
MarginSampling = [
    [0.3667, 0.424, 0.4677, 0.4983, 0.5232, 0.5456, 0.5544, 0.5555, 0.5956, 0.5744, 0.6004],
    [0.4341, 0.4624, 0.4829, 0.5065, 0.5364, 0.5439, 0.5643, 0.5726, 0.5734, 0.6025, 0.6093],
    [0.5152, 0.5251, 0.5537, 0.5592, 0.5761, 0.5756, 0.5851, 0.6034, 0.6209, 0.6237, 0.6355],
    [0.593, 0.5918, 0.5941, 0.6062, 0.6222, 0.619, 0.6311, 0.6359, 0.6332, 0.6474, 0.6608]
]
EntropySampling = [
    [0.3667, 0.4083, 0.4471, 0.4524, 0.4988, 0.5531, 0.54, 0.5594, 0.5856, 0.5843, 0.6087],
    [0.4341, 0.4613, 0.4956, 0.5015, 0.5283, 0.5617, 0.5672, 0.5807, 0.5899, 0.5943, 0.5936],
    [0.5152, 0.5349, 0.5447, 0.5652, 0.5609, 0.5759, 0.5991, 0.605, 0.5991, 0.6172, 0.6213],
    [0.593, 0.5946, 0.598, 0.6067, 0.6205, 0.619, 0.6441, 0.6293, 0.6287, 0.6478, 0.6624]
]
full_learning = [
    [0.3667, 0.4341, 0.4691, 0.5, 0.5152, 0.5419, 0.5523, 0.5705, 0.5778, 0.593, 0.5963],
    [0.4341, 0.4691, 0.5, 0.5152, 0.5419, 0.5523, 0.5705, 0.5778, 0.593, 0.5963, 0.5967],
    [0.5152, 0.5419, 0.5523, 0.5705, 0.5778, 0.593, 0.5963, 0.5967, 0.6098, 0.6072, 0.6177],
    [0.593, 0.5963, 0.5967, 0.6098, 0.6072, 0.6177, 0.6204, 0.6283, 0.6408, 0.6403, 0.648]
]

sampling_ans = {'RandomSampling': RandomSampling, 'LeastConfidence': LeastConfidence, 'MarginSampling': MarginSampling, 'EntropySampling': EntropySampling}
init_ans = dict([(key, [i[j] for i in sampling_ans.values()]) for j, key in enumerate([1000, 2000, 5000, 10000])])

# 计算主动学习和全量学习准确率提升率
incremental_rates = dict([(key, [[(incre - full)/full for incre, full in zip(value[i], full_learning[i])] for i in range(4)]) for key, value in sampling_ans.items()])
incremental_rates_by_init = dict([(key, [i[j] for i in incremental_rates.values()]) for j, key in enumerate([1000, 2000, 5000, 10000])])

global_max = max([max([max(j) for j in i]) for i in incremental_rates.values()])
global_min = min([min([min(j) for j in i]) for i in incremental_rates.values()])
global_abs_max = max(abs(global_max), abs(global_min))
global_abs_min = min([min([min([abs(k) for k in j]) for j in i]) for i in incremental_rates.values()])


def draw_ans_by_sampling_way(used_sampling):
    sampling_name = used_sampling
    used_sampling = sampling_ans[sampling_name]
    incremental_rate = incremental_rates[sampling_name]

    # 计算标记点大小
    min_proj, max_proj = 40, 400
    marker_size = [[(abs(i) - global_abs_min) / (global_abs_max - global_abs_min) * (max_proj - min_proj) + min_proj for i in incremental_rate[j]] for j in range(4)]  # 映射到40-400
    max_size = max(max(i) for i in marker_size)
    min_size = min(min(i) for i in marker_size)

    # 绘制图表
    plt.figure(figsize=(10, 6))

    colors = ['#1F77B4', '#2CA02C', '#9467BD', '#E377C2']

    # 绘制增量学习效果
    pos_max_size = 0
    neg_max_size = 0
    for i in range(4):
        plt.plot(increased_labeled, used_sampling[i], label=f'初始标记样本-{init_labeled[i]}', marker='.', color=colors[i], zorder=1)
        plt.ylim((0.31, 0.68))
        plt.yticks(np.arange(0.35, 0.68, 0.05))
        pos_label = []
        pos_sampling = []
        pos_marker_size = []
        neg_label = []
        neg_sampling = []
        neg_marker_size = []
        for j, incre in enumerate(incremental_rate[i]):
            if incre >= 0:
                pos_label.append(increased_labeled[j])
                pos_sampling.append(used_sampling[i][j])
                pos_marker_size.append(marker_size[i][j])
            else:
                neg_label.append(increased_labeled[j])
                neg_sampling.append(used_sampling[i][j])
                neg_marker_size.append(marker_size[i][j])
        plt.scatter(pos_label, pos_sampling,  marker='o', color=colors[i], s=pos_marker_size, zorder=2)
        plt.scatter(neg_label, neg_sampling,  marker='v', color=colors[i], s=neg_marker_size, zorder=2)
        if len(pos_marker_size) > 0:
            pos_max_size = max(max(pos_marker_size), pos_max_size)
        if len(neg_marker_size) > 0:
            neg_max_size = max(max(neg_marker_size), neg_max_size)

    # 添加图例
    l1 = plt.legend(fontsize=15, markerscale=2, loc='lower right')
    real_max = max([max(i) for i in incremental_rate])
    real_min = min([min(i) for i in incremental_rate])
    color = '#E38484'
    if real_min >= 0:
        scatter_sizes = [plt.scatter([], [],  marker='o', color=color, s=max_size, label=f'{round(real_max * 100, 2)}%'),
                         plt.scatter([], [],  marker='o', color=color, s=min_size, label=f'{round(real_min * 100, 2)}%')]
    else:
        scatter_sizes = [plt.scatter([], [],  marker='o', color=color, s=pos_max_size, label=f'{round(real_max * 100, 2)}%'),
                         plt.scatter([], [],  marker='o', color=color, s=min_proj, label='0 %'),
                         plt.scatter([], [], marker='v', color=color,
                                     s=neg_max_size,
                                     label=f'{round(real_min * 100, 2)}%')]
    l2 = plt.legend(handles=scatter_sizes, fontsize=15, loc='upper left', columnspacing=-0.1, handletextpad=-0.1, ncols=3)
    plt.gca().add_artist(l1)

    # 添加标题和标签
    plt.title(sampling_name)
    plt.xlabel('增量学习样本数')
    plt.ylabel('准确率')

    # 显示图表
    plt.grid(True)
    plt.show()


def draw_ans_by_init_label(used_init):
    init_value = used_init
    used_init = init_ans[init_value]
    incremental_rate = incremental_rates_by_init[init_value]

    # 计算标记点大小
    min_proj, max_proj = 40, 400
    marker_size = [[(abs(i) - global_abs_min) / (global_abs_max - global_abs_min) * (max_proj - min_proj) + min_proj for i in incremental_rate[j]] for j in range(4)]  # 映射到40-400
    max_size = max(max(i) for i in marker_size)
    min_size = min(min(i) for i in marker_size)

    # 绘制图表
    plt.figure(figsize=(10, 6))

    colors = ['#1F77B4', '#2CA02C', '#9467BD', '#E377C2']

    # 绘制增量学习效果
    pos_max_size = 0
    neg_max_size = 0
    labels = list(sampling_ans.keys())
    for i in range(4):
        plt.plot(increased_labeled, used_init[i], label=labels[i], marker='.', color=colors[i], zorder=1)
        plt.ylim((0.31, 0.68))
        plt.yticks(np.arange(0.35, 0.68, 0.05))
        pos_label = []
        pos_init = []
        pos_marker_size = []
        neg_label = []
        neg_init = []
        neg_marker_size = []
        for j, incre in enumerate(incremental_rate[i]):
            if incre >= 0:
                pos_label.append(increased_labeled[j])
                pos_init.append(used_init[i][j])
                pos_marker_size.append(marker_size[i][j])
            else:
                neg_label.append(increased_labeled[j])
                neg_init.append(used_init[i][j])
                neg_marker_size.append(marker_size[i][j])
        plt.scatter(pos_label, pos_init,  marker='o', color=colors[i], s=pos_marker_size, zorder=2, alpha=1)
        plt.scatter(neg_label, neg_init,  marker='v', color=colors[i], s=neg_marker_size, zorder=2, alpha=1)
        if len(pos_marker_size) > 0:
            pos_max_size = max(max(pos_marker_size), pos_max_size)
        if len(neg_marker_size) > 0:
            neg_max_size = max(max(neg_marker_size), neg_max_size)

    # 添加图例
    l1 = plt.legend(fontsize=15, markerscale=2, loc='lower right')
    real_max = max([max(i) for i in incremental_rate])
    real_min = min([min(i) for i in incremental_rate])
    color = '#E38484'
    if real_min >= 0:
        scatter_sizes = [plt.scatter([], [],  marker='o', color=color, s=max_size, label=f'{round(real_max * 100, 2)}%'),
                         plt.scatter([], [],  marker='o', color=color, s=min_size, label=f'{round(real_min * 100, 2)}%')]
    else:
        scatter_sizes = [plt.scatter([], [],  marker='o', color=color, s=pos_max_size, label=f'{round(real_max * 100, 2)}%'),
                         plt.scatter([], [],  marker='o', color=color, s=min_proj, label='0 %'),
                         plt.scatter([], [], marker='v', color=color,
                                     s=neg_max_size,
                                     label=f'{round(real_min * 100, 2)}%')]
    l2 = plt.legend(handles=scatter_sizes, fontsize=15, loc='upper left', columnspacing=-0.1, handletextpad=-0.1, ncols=3)
    plt.gca().add_artist(l1)

    # 添加标题和标签
    plt.title(f'初始标记样本-{init_value}')
    plt.xlabel('增量学习样本数')
    plt.ylabel('准确率')

    # 显示图表
    plt.grid(True)
    plt.show()




if __name__ == '__main__':
    for used_sampling in sampling_ans.keys():
        draw_ans_by_sampling_way(used_sampling)

    for used_init in init_ans.keys():
        draw_ans_by_init_label(used_init)
