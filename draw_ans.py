import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
plt.rcParams.update({"font.size": 18})

# 数据
init_labeled = [1000, 2000, 5000, 10000]
increased_labeled = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
RandomSampling = [
    [0.3558, 0.4026, 0.4364, 0.4674, 0.4062, 0.5172, 0.5522, 0.56, 0.5206, 0.5693, 0.5843],
    [0.3913, 0.46, 0.4784, 0.4665, 0.529, 0.5383, 0.5284, 0.5603, 0.5638, 0.5823, 0.5895],
    [0.5023, 0.5356, 0.5373, 0.559, 0.5356, 0.5624, 0.5999, 0.5892, 0.5373, 0.6048, 0.6134],
    [0.5685, 0.5763, 0.5696, 0.4575, 0.6111, 0.616, 0.6129, 0.6214, 0.6024, 0.6322, 0.6302]
]
LeastConfidence = [
    [0.3558, 0.2969, 0.4149, 0.4554, 0.475, 0.5162, 0.4549, 0.5528, 0.5673, 0.5494, 0.5742],
    [0.3913, 0.4433, 0.4911, 0.429, 0.5008, 0.522, 0.5478, 0.5442, 0.4878, 0.5679, 0.5653],
    [0.5023, 0.5162, 0.4728, 0.5461, 0.5325, 0.5733, 0.5866, 0.5677, 0.5001, 0.5809, 0.5778],
    [0.5685, 0.5484, 0.5794, 0.5876, 0.5962, 0.6189, 0.6285, 0.6127, 0.5789, 0.639, 0.62]
]
MarginSampling = [
    [0.3558, 0.3686, 0.4483, 0.4886, 0.446, 0.5098, 0.5272, 0.5508, 0.5541, 0.5282, 0.5849],
    [0.3913, 0.4453, 0.4678, 0.4841, 0.5193, 0.5321, 0.554, 0.5574, 0.4772, 0.5856, 0.5895],
    [0.5016, 0.5094, 0.5, 0.5524, 0.5665, 0.5373, 0.5836, 0.5831, 0.4821, 0.6113, 0.5956],
    [0.5265, 0.6028, 0.5706, 0.5125, 0.612, 0.6097, 0.6122, 0.6006, 0.6059, 0.6349, 0.6359]
]
EntropySampling = [
    [0.3662, 0.3469, 0.4445, 0.4574, 0.2757, 0.4755, 0.4924, 0.5437, 0.5555, 0.5367, 0.5803],
    [0.3885, 0.4496, 0.4787, 0.242, 0.5042, 0.5071, 0.5547, 0.5392, 0.514, 0.5544, 0.546],
    [0.5016, 0.525, 0.5016, 0.5177, 0.549, 0.546, 0.5883, 0.5645, 0.5851, 0.5719, 0.59],
    [0.5265, 0.5954, 0.5985, 0.5204, 0.6006, 0.6028, 0.6204, 0.6222, 0.5549, 0.6271, 0.6067]
]
full_learning = [
    [0.3662, 0.3885, 0.4547, 0.4846, 0.5016, 0.5141, 0.4857, 0.5595, 0.5539, 0.5265, 0.5711],
    [0.3885, 0.4547, 0.4846, 0.5016, 0.5141, 0.4857, 0.5595, 0.5539, 0.5265, 0.5711, 0.5626],
    [0.5016, 0.5141, 0.4857, 0.5595, 0.5539, 0.5265, 0.5711, 0.5626, 0.4353, 0.606, 0.5983],
    [0.5265, 0.5711, 0.5626, 0.4353, 0.606, 0.5983, 0.6147, 0.6316, 0.6195, 0.6409, 0.6255]
]

sampling_ans = {'RandomSampling': RandomSampling, 'LeastConfidence': LeastConfidence, 'MarginSampling': MarginSampling, 'EntropySampling': EntropySampling}


def draw_ans(used_sampling):
    sampling_name = used_sampling
    used_sampling = sampling_ans[used_sampling]
    # 计算主动学习和全量学习准确率提升率
    incremental_rate = [[(incre - full)/full for incre, full in zip(used_sampling[i], full_learning[i])] for i in range(4)]

    # 计算标记点大小
    maxi = max([max([abs(j) for j in i]) for i in incremental_rate])
    mini = min([min([abs(j) for j in i]) for i in incremental_rate])
    min_proj, max_proj = 40, 400
    marker_size = [[(abs(i) - mini) / (maxi - mini) * (max_proj - min_proj) + min_proj for i in incremental_rate[j]] for j in range(4)]  # 映射到40-400


    # 绘制图表
    plt.figure(figsize=(10, 6))

    colors = ['#1F77B4', '#2CA02C', '#9467BD', '#E377C2']

    # 绘制增量学习效果
    pos_max_size = 0
    neg_max_size = 0
    for i in range(4):
        plt.plot(increased_labeled, used_sampling[i], label=f'初始标记样本-{init_labeled[i]}', marker='.', color=colors[i], zorder=1)
        plt.ylim((0.21, 0.68))
        plt.yticks(np.arange(0.25, 0.68, 0.05))
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
    color = '#EB8A8A'
    if real_min >= 0:
        scatter_sizes = [plt.scatter([], [],  marker='o', color=color, s=max_proj, label=f'{round(real_max * 100, 2)}%'),
                         plt.scatter([], [],  marker='o', color=color, s=min_proj, label=f'{round(real_min * 100, 2)}%')]
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


for used_sampling in sampling_ans.keys():
    draw_ans(used_sampling)
