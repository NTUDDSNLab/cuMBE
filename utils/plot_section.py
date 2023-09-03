import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

algorithms = ['noRS', 'noES', 'noWS', 'cuMBE']
sections = ['L\' Construction', 'Maximality Checking', 'Maximality Expansion', 'Candidate Selection', 'Subtree Fetching', 'Work-stealing', 'Others']
ingredients = [[3], [4], [5], [7], [8], [2, 9], 0]
datasets = [
    'DBLP-author'         ,
    'DBpedia_locations'   ,
    'Marvel'              ,
    'YouTube'             ,
    'IMDB-actor'          ,
    'stackoverflow'       ,
    'BookCrossing'        ,
    'corporate-leadership',
    'movielens-t-i'       ,
    'movielens-u-i'       ,
    'movielens-u-t'       ,
    'UCforum'             ,
    'Unicode'             ,
]

execution_times = [ # default result
#   [     'noRS',      'noES',     'noWS',    'cuMBE', ] (seconds)
    [3246.625488,  228.884415, 222.518433, 228.360321, ], # 'DBLP-author'
    [   1.364625,    0.112622,   0.095693,   0.108761, ], # 'DBpedia_locations'
    [   0.210414,    0.095391,   0.128242,   0.070486, ], # 'Marvel'
    [  28.023441,    7.419772,   3.264178,   2.213176, ], # 'YouTube'
    [ 239.366287,  172.004272,  31.250929,  31.383694, ], # 'IMDB-actor'
    [2914.852051, 1080.250854, 490.996796, 350.133240, ], # 'stackoverflow'
    [7459.787598, 1385.010254, 719.730896, 620.670532, ], # 'BookCrossing'
    [   0.011273,    0.011235,   0.000382,   0.011193, ], # 'corporate-leadership'
    [   1.374244,    1.651912,   0.516421,   0.474633, ], # 'movielens-t-i'
    [   2.902356,    1.014329,   2.395441,   0.856843, ], # 'movielens-u-i'
    [   0.283220,    0.170586,   0.173466,   0.098959, ], # 'movielens-u-t'
    [   0.017888,    0.022206,   0.010777,   0.016384, ], # 'UCforum'
    [   0.008725,    0.004401,   0.000971,   0.004443, ], # 'Unicode'
]

# 每個 code section 在不同 dataset 上的執行時間比例 (單位: %)
# log file 中的 section time 有 (0) ~ (10) 共 11 個
# execution_sections 分別填入 [(7), (3), (4), (5), (8), (2)+(9), 0 或是 (0)+(1)+(6)+(10)]
execution_sections = [ # default result
    [ # 'DBLP-author'
    [0.05  , 0.04  , 98.60  , 0.03  ,  1.14  , 0.08  +0.05  , 0],   # 'noRS'
    [1.4681, 0.7620,  0.6922, 2.0126, 92.1095, 1.0998+0.7151, 0],   # 'noES'
    [1.0898, 0.6984,  0.7132, 1.0480, 95.4779, 0.0000+0.0000, 0],   # 'noWS'
    [1.4312, 0.7497,  0.7045, 1.0184, 93.1125, 1.1456+0.7063, 0],   # 'cuMBE'
    ],
    [ # 'DBpedia_locations'
    [0.49  , 1.02  , 91.85  , 0.37  ,  4.50  , 0.51  +0.43  , 0],   # 'noRS'
    [4.4221, 1.3479,  1.1936, 3.2664, 69.0621, 5.9185+4.7826, 0],   # 'noES'
    [3.6986, 1.2905,  1.2545, 1.9973, 80.6491, 0.0000+0.0000, 0],   # 'noWS'
    [4.2299, 1.3109,  1.1930, 1.9586, 71.1094, 5.6529+4.6835, 0],   # 'cuMBE'
    ],
    [ # 'Marvel'
    [ 4.05  , 49.49  , 9.34  , 10.43  , 1.65  , 0.09  + 8.12  , 0], # 'noRS'
    [ 8.2695, 13.4791, 4.7212, 40.0245, 3.4489, 0.1980+19.1907, 0], # 'noES'
    [ 3.8819,  5.3716, 2.1326, 10.5046, 2.2633, 0.0000+ 0.0000, 0], # 'noWS'
    [10.0612, 17.5447, 6.3395, 26.2770, 4.6675, 0.2907+22.2357, 0], # 'cuMBE'
    ],
    [ # 'YouTube'
    [1.45  , 82.81  , 4.27  ,  6.56  , 0.12  , 0.01  +0.06  , 0],   # 'noRS'
    [3.3117, 16.9917, 2.3740, 74.2871, 0.3356, 0.0210+0.2757, 0],   # 'noES'
    [3.6901, 25.0514, 3.2608, 18.1081, 0.6698, 0.0000+0.0000, 0],   # 'noWS'
    [7.2118, 46.7550, 6.8545, 30.6956, 1.0450, 0.0647+0.7781, 0],   # 'cuMBE'
    ],
    [ # 'IMDB-actor'
    [1.01  , 38.19  , 34.57  , 24.31  , 0.59  , 0.07  +0.01  , 0],  # 'noRS'
    [0.7180,  5.5017,  0.4118, 91.5962, 0.9924, 0.1133+0.0148, 0],  # 'noES'
    [2.0880, 31.1734,  2.4466, 54.0935, 6.6022, 0.0000+0.0000, 0],  # 'noWS'
    [2.1048, 31.2926,  2.4596, 53.9148, 6.5863, 0.2955+0.0734, 0],  # 'cuMBE'
    ],
    [ # 'stackoverflow'
    [0.30  , 87.70  , 0.75  ,  9.43  , 0.01  , 0.00  +0.00  , 0],   # 'noRS'
    [0.3721, 23.2200, 0.4227, 75.6615, 0.0307, 0.0012+0.0064, 0],   # 'noES'
    [0.5363, 48.2071, 0.7437, 16.5932, 0.0689, 0.0000+0.0000, 0],   # 'noWS'
    [0.8324, 72.8362, 1.3348, 24.0432, 0.0978, 0.0051+0.0149, 0],   # 'cuMBE'
    ],
    [ # 'BookCrossing'
    [0.84  , 88.66  , 1.89  ,  6.34  , 0.00  , 0.00  +0.00  , 0],   # 'noRS'
    [2.7635, 22.9079, 1.7595, 71.8016, 0.0308, 0.0009+0.0034, 0],   # 'noES'
    [3.8930, 34.6451, 2.3583, 29.2074, 0.0586, 0.0000+0.0000, 0],   # 'noWS'
    [5.7208, 50.0359, 3.8974, 38.7392, 0.0683, 0.0019+0.0066, 0],   # 'cuMBE'
    ],
    [ # 'corporate-leadership'
    [0.01  , 0.01  , 0.00  , 0.00  , 0.01  , 0.04  +98.78  , 0],    # 'noRS'
    [0.0064, 0.0068, 0.0042, 0.0043, 0.0068, 0.0514+98.8207, 0],    # 'noES'
    [0.4093, 0.4299, 0.3037, 0.6083, 0.1797, 0.0000+ 0.0000, 0],    # 'noWS'
    [0.0063, 0.0068, 0.0043, 0.0034, 0.0070, 0.0356+98.9411, 0],    # 'cuMBE'
    ],
    [ # 'movielens-t-i'
    [1.63  , 64.06  , 1.97  , 22.63  , 0.32  , 0.03  +1.57  , 0],   # 'noRS'
    [1.2925, 12.7608, 0.8723, 81.1905, 0.2342, 0.0221+1.7206, 0],   # 'noES'
    [2.6026, 34.9311, 2.5853, 34.4644, 0.6599, 0.0000+0.0000, 0],   # 'noWS'
    [3.0479, 43.7684, 3.1723, 39.2280, 0.8009, 0.0721+4.9093, 0],   # 'cuMBE'
    ],
    [ # 'movielens-u-i'
    [3.21  , 64.52  , 7.82  ,  9.62  , 0.08  , 0.01  +1.05  , 0],   # 'noRS'
    [8.2207, 26.5495, 7.0676, 46.3403, 0.2163, 0.0200+3.0521, 0],   # 'noES'
    [2.0361,  5.0361, 1.4298,  5.9181, 0.0672, 0.0000+0.0000, 0],   # 'noWS'
    [9.7158, 35.0714, 9.2754, 32.2851, 0.2713, 0.0190+3.4839, 0],   # 'cuMBE'
    ],
    [ # 'movielens-u-t'
    [2.36  , 57.47  , 7.63  , 11.50  , 0.71  , 0.06  + 7.20  , 0],  # 'noRS'
    [4.1489, 15.9580, 3.2605, 55.9092, 1.1629, 0.1227+14.2897, 0],  # 'noES'
    [2.4410,  9.5643, 2.0759, 12.6606, 0.9352, 0.0000+ 0.0000, 0],  # 'noWS'
    [5.7288, 26.4373, 5.7569, 29.3355, 1.9426, 0.1939+22.2785, 0],  # 'cuMBE'
    ],
    [ # 'UCforum'
    [2.45  , 8.33  , 1.81  ,  8.75  , 1.00  , 0.12  +56.15  , 0],   # 'noRS'
    [1.8860, 3.6472, 1.2323, 18.8160, 0.7980, 0.0836+51.4680, 0],   # 'noES'
    [3.5351, 6.9610, 2.3323, 12.2123, 1.1627, 0.0000+ 0.0000, 0],   # 'noWS'
    [2.4748, 4.8261, 1.6326,  8.0786, 1.0694, 0.1175+64.0105, 0],   # 'cuMBE'
    ],
    [ # 'Unicode'
    [0.29  , 0.93  , 0.17  , 0.02  , 1.33  , 0.27  +94.50  , 0],    # 'noRS'
    [0.6042, 1.4127, 0.2902, 0.4647, 2.2220, 0.2980+90.1123, 0],    # 'noES'
    [2.7597, 7.4825, 1.5464, 5.8730, 4.9493, 0.0000+ 0.0000, 0],    # 'noWS'
    [0.5892, 1.3811, 0.2941, 0.0455, 2.3729, 0.3743+91.2296, 0],    # 'cuMBE'
    ],
]

# broken_datasets: 所有 dataset 對應 broken_location/length/target 的 index (-1 表示無 broken axis)
# broken_location: broken point 實際位置 (在 subplot 上作圖的位置)
# broken_length  : broken point 所省略的長度
# broken_target  : 所有 algorithm 對應的 broken point 分別位於哪個 section，
#                  2 表示 broken point 位於 sections[2]，-1 表示此 algorithm 無 broken point
broken_datasets = [0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
broken_location = [350, 0.175]
broken_length = [2700, 1.05]
broken_target = [
    [sections.index('Maximality Expansion'), -1, -1, -1],
    [sections.index('Maximality Expansion'), -1, -1, -1],
]

# 從 result file 取得 runtime 相關數據
def get_from_result_files():
    execution_sections = [[[0 for x in sections] for y in algorithms] for z in datasets]
    execution_times    = [[0                     for y in algorithms] for z in datasets]
    print(np.shape(execution_sections))
    print(np.shape(execution_times))
    for dataset_index in range(len(datasets)):
        for algorithm_index in range(len(algorithms)):
            result = open("result/{}_{}_section".format(datasets[dataset_index], algorithms[algorithm_index]), "r")
            lines = result.readlines()
            for line in lines:
                # 取得 time percentage，存於 execution_sections
                if 'time percentage:' in line:
                    time_percentages = [float(x) for x in line.replace('time percentage:', '').split()]
                    new_sections = [0 for x in sections]
                    for section_index in range(len(sections)):
                        if ingredients[section_index] == 0:
                            continue
                        for ingr in ingredients[section_index]:
                            new_sections[section_index] += time_percentages[ingr]
                    execution_sections[dataset_index][algorithm_index] = new_sections
                # 取得 runtime，存於 execution_times
                elif 'runtime (s):' in line:
                    execution_times[dataset_index][algorithm_index] = float(line.replace('runtime (s):', '').strip())

# 在 subplot 上繪製 broken point (白色平行四邊形*1 + 黑色斜線*2)，覆蓋在 bar 上
def cut_at(cut_x, cut_y, cut_w, cut_h):
    x = [cut_x - 0.50*cut_w, cut_x + 0.50*cut_w, cut_x + 0.50*cut_w, cut_x - 0.50*cut_w]
    y = [cut_y - 0.75*cut_h, cut_y - 0.25*cut_h, cut_y + 0.75*cut_h, cut_y + 0.25*cut_h]
    ax.add_patch(patches.Polygon(xy=list(zip(x,y)), fill=True, color='w'))
    ax.plot(x[  :2], y[  :2], color='black')
    ax.plot(x[-2: ], y[-2: ], color='black')

# 各 section 顏色
# colors = ['#F43545', '#FF8901', '#00BA71', '#00C2DE', '#00418D', '#5F2879', '#292421']
# colors = ['#DC6464', '#DCDC64', '#64DC64', '#64DCDC', '#6464DC', '#DC64DC', '#606060']
# colors = ['#E84C54', '#EDB232', '#64DC64', '#64DCDC', '#6464DC', '#9D46AA', '#606060']
colors = ['#ef1010', '#eeb711', '#0dbf0d', '#0dbfbf', '#1111ee', '#b50db5', '#606060']
# colors = ['#b50db5', '#d72828', '#eeb711', '#0dbf0d', '#0fd7d7', '#1111ee', '#606060']

# 整張 figure 直向橫向有幾張 subplot (直, 橫)
subplot_size = (2, 7)
# 設定整張 figure 的 size
fig = plt.figure(figsize=(subplot_size[1]*(1 + len(algorithms)*0.5), subplot_size[0]*3.825))


# 從 result file 取得 runtime 相關數據
get_from_result_files()

# 轉置 execution_sections 的軸 1、軸 2，以供 ax.bar() 作圖
# shape: (len(datasets), len(algorithms), len(sections)) ---> (len(datasets), len(sections), len(algorithms))
for dataset_index in range(len(datasets)):
    execution_sections[dataset_index] = list(map(list, zip(*execution_sections[dataset_index])))


# 迭代不同 subplot
for i in range(subplot_size[0]):
    for j in range(subplot_size[1]):


        # 計算 dataset 編號
        dataset_index = subplot_size[1] * i + j
        if dataset_index >= len(datasets):
            break
        print("Dataset: {}".format(datasets[dataset_index]))

        # 若未填寫(填寫 0)，計算此 dataset 每個 algorithm 的 "Other" section 的比例 (%)
        for algorithm_index in range(len(algorithms)):
            ratio_sum = 0
            if execution_sections[dataset_index][-1][algorithm_index] == 0:
                for section_index in range(len(sections)):
                    ratio_sum += execution_sections[dataset_index][section_index][algorithm_index]
                execution_sections[dataset_index][-1][algorithm_index] = 100 - ratio_sum

        # 建立此 subplot
        ax = plt.subplot2grid(subplot_size, (i,j))
        
        bar_width = 0.5
        x_index = np.arange(len(algorithms))

        # 此 dataset 的 execution_times，shape: (len(sections), len(algorithms))
        dataset_execution_times = execution_times[dataset_index]
        # 此 dataset 的每個 algorithm 的 bottom_value，繪製 bar 時使用
        bottom_values = np.zeros(len(algorithms))
        # # 此 dataset 的 broken_dataset 的 index
        # broken_index = broken_datasets[dataset_index] # fake broken axis


        # 迭代不同 section
        for k, section_ratios in enumerate(execution_sections[dataset_index]):

            # 計算所有 algorithm 的此 section 的執行時間
            bar_lengths = [(section_ratios[x] * dataset_execution_times[x] / 100) for x in range(len(algorithms))]
            print("    Section: {}".format(sections[k]))
            print("        {}".format(bar_lengths))
            # # 如果此 dataset 的此 algorithm 的此 section 需要做 broken axis 處理，減少此 bar 長度
            # if broken_index != -1:
            #     for x in range(len(algorithms)):
            #         if k == broken_target[broken_index][x]:
            #             bar_lengths[x] -= broken_length[broken_index]
            
            # 繪製所有 algorithm 的此 section 的執行時間的 bar
            ax.bar(x_index, bar_lengths, bar_width, color=colors[k], label=sections[k], bottom=bottom_values, edgecolor='black')
            # 下個 section 的 bar 的起始點
            bottom_values += bar_lengths
        

        # # 如果此 dataset 需要做 broken axis 處理
        # if broken_index != -1:

        #     plt.draw() # 使 ax.get_yticklabels() 可用
        #     # 獲取預設的 yticks 的數值
        #     broken_ylabels = [item.get_text() for item in ax.get_yticklabels()]
        #     # 獲取預設的 yticks 的小數點後精度
        #     precision = 0 if broken_ylabels[0].rfind('.') == -1 else len(broken_ylabels[0]) - broken_ylabels[0].rfind('.') - 1

        #     # 修正 broken point 以上的 yticks 的 label 數值
        #     for y in range(len(broken_ylabels)):
        #         if (float(broken_ylabels[y]) >= broken_location[broken_index]):
        #             new_value = float(broken_ylabels[y]) + broken_length[broken_index]
        #             broken_ylabels[y] = f"{new_value:.{precision}f}"
        #     # yticks 的 label 設為修改後的版本
        #     ax.set_yticklabels(broken_ylabels)

        #     # 若此 algorithm 有 broken point，在 subplot 的對應位置繪製
        #     for x in range(len(algorithms)):
        #         if bottom_values[x] >= broken_location[broken_index]:
        #             cut_at(x, broken_location[broken_index], bar_width*1.2, ax.get_ylim()[1]*0.025)
        

        # 微調 subplot 性質
        ax.set_title(datasets[dataset_index])
        ax.set_xticks(x_index, algorithms)
        ax.set_xlim((-1+bar_width*0.5, len(algorithms)-bar_width*0.5))
        if j == 0: # 僅左側 subplot 繪製 y 軸標題
            ax.set_ylabel('Execution Time (seconds)')


# 繪製 legend 於整張 figure 右下角 (以 bbox_to_anchor 微調位置)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, fontsize=12, loc='lower right', bbox_to_anchor=(0.987,0.039))

plt.tight_layout()
plt.show()

plt.savefig('section.png', dpi=640, bbox_inches='tight')
plt.savefig('section.eps', format='eps', bbox_inches='tight')

exit()