import argparse
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import seaborn as sns
import glob
import dask.dataframe as dd
import matplotlib.patches as plt_patches
import matplotlib.lines as plt_lines

from numpy import loadtxt

def cut_at(cut_x, cut_y, cut_w, cut_h):
    x = [cut_x - 0.50*cut_w, cut_x + 0.50*cut_w, cut_x + 0.50*cut_w, cut_x - 0.50*cut_w]
    y = [cut_y / (cut_h**0.75), cut_y / (cut_h**0.25), cut_y * (cut_h**0.75), cut_y * (cut_h**0.25)]
    ax.add_patch(plt_patches.Polygon(xy=list(zip(x,y)), fill=True, color='w', zorder=10))
    ax.plot(x[  :2], y[  :2], color='black', zorder=11)
    ax.plot(x[-2: ], y[-2: ], color='black', zorder=11)

# Adjust the widths of a seaborn-generated boxplot.
def adjust_boxplots(g, width_fac=1, shift_dist=0):

    # iterating through Axes instances
    for ax in g.axes:
        # iterating through axes artists:
        for c in ax.get_children():

            c_type = 'Patch'   if isinstance(c, plt_patches.PathPatch) \
                else 'Polygon' if isinstance(c, plt_patches.Polygon) \
                else 'Line2D'  if isinstance(c, plt_lines.Line2D) \
                else 'Other'

            # searching for PathPatches and Line2Ds
            if c_type != 'Other':
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices[:-1] if c_type == 'Patch' else p.vertices
                xmin  = np.min(verts[:, 0])
                xmax  = np.max(verts[:, 0])
                xmid  = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # Patch : setting new width of box
                # Line2D: setting new width of Ts and median lines
                shift_x  = shift_dist if xmid - round(xmid) > 0 else -shift_dist
                xmin_new = xmid - width_fac * xhalf + shift_x
                xmax_new = xmid + width_fac * xhalf + shift_x
                verts[verts[:, 0] == xmin, 0] = xmin_new
                verts[verts[:, 0] == xmax, 0] = xmax_new


datas = [
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
# fig, ax = plt.subplots(figsize=(30, 8))
fig, ax = plt.subplots(figsize=(22.5, 6))

data_exts = [(-100, +100) for x in range(len(datas))]
data_stdevs_WS   = []
data_stdevs_noWS = []
df = pd.DataFrame(columns=['dataset', 'clock', 'work_stealing'])
for data in datas:
    ori_br = open("result/{}_cuMBE_variance".format(data), "r")
    lines = ori_br.readlines()
    del lines[0:5]
    del lines[-1]
    del lines[-1]
    res = [eval(i) for i in lines]
    sum = 0
    for num in res:
        sum += num/len(res)

    data_stdevs_WS.append(np.std(res)/sum)
    for line in res:
        df = df.append({'dataset':'{}'.format(data), 'clock':line/sum, 'work_stealing': 'WS'}, ignore_index=True)

for data in datas:
    ori_br = open("result/{}_noWS_variance".format(data), "r")
    lines = ori_br.readlines()
    del lines[0:5]
    del lines[-1]
    del lines[-1]
    res = [eval(i) for i in lines]
    sum = 0
    for num in res:
        sum += num/len(res)

    data_exts[datas.index(data)] = (min(res)/sum, max(res)/sum)
    data_stdevs_noWS.append(np.std(res)/sum)
    if data == 'Unicode':
        for line in res:
            new_clock = min(res)/sum/0.0625*0.1625 if line/sum < 0.1625 else line/sum
            df = df.append({'dataset':'{}'.format(data), 'clock':new_clock, 'work_stealing': 'noWS'}, ignore_index=True)
    elif data == 'corporate-leadership':
        for line in res:
            new_clock = max(res)/sum/16*8 if line/sum > 8 else line/sum
            df = df.append({'dataset':'{}'.format(data), 'clock':new_clock, 'work_stealing': 'noWS'}, ignore_index=True)
    else:
        for line in res:
            df = df.append({'dataset':'{}'.format(data), 'clock':line/sum, 'work_stealing': 'noWS'}, ignore_index=True)

# print(data_exts)
print('[Standard Deviation]')
# print(data_stdevs_WS)
print('  WS:', min(data_stdevs_WS), '~', max(data_stdevs_WS))
# print(data_stdevs_noWS)
print('noWS:', min(data_stdevs_noWS), '~', max(data_stdevs_noWS))

# my_fig = sns.boxplot(x="dataset", y="clock", hue="work_stealing", data=df, showfliers=False, width=0.6)
# cut_at(len(datas)-1 + 0.15,0.17,0.2,1.08)
my_fig = sns.boxplot(x="dataset", y="clock", hue="work_stealing", data=df, showfliers=False, width=0.4, whis=float("inf"))
cut_at(datas.index('Unicode') + 0.1, 0.17, 0.16, 1.06)
# cut_at(datas.index('corporate-leadership') + 0.1, (data_exts[datas.index('corporate-leadership')][1]/16*8*data_exts[datas.index('movielens-u-i')][1])**0.5, 0.16, 1.058)
cut_at(datas.index('corporate-leadership') + 0.1, (21/16*8*6)**0.5, 0.16, 1.06)
adjust_boxplots(fig, shift_dist=0.075)

xticklabel_objects = my_fig.set_xticklabels(my_fig.get_xticklabels(), fontsize=9)
xticklabel_objects[datas.index('corporate-leadership')].set_fontsize(7.8)
plt.grid(axis='y', linestyle='--', color='grey')

# y_ticks = [2**i for i in range(-3, int(np.log2(max(df['clock'])))+1)]
y_ticks = [2**i for i in range(-3, 5)]
y_ticks[0] = 0.05/0.0625*0.1625
y_ticks[-1] = 21/16*8
y_ticks[-2] = 6
plt.xticks()

plt.yscale('log', base=2)
#y_ticks = [2**i for i in range(-4, int(np.log2(max(df['clock'])))+1)]
#y_labels =  [f'{tick:.3f}' for tick in y_ticks] # 将标签格式化为小数形式
#y_labels = [f'{int(tick)}' if tick == int(tick) else f'{tick:.3f}' for tick in y_ticks]
y_labels = [f'{tick:.3f}'.rstrip('0').rstrip('.') if tick != int(tick) else f'{int(tick)}' for tick in y_ticks]
y_labels[0] = 0.05
y_labels[-1] = 21

plt.yticks(y_ticks, y_labels)


ax.set_xticks(np.arange(-0.5, len(datas)+0.5, 1), minor=True)
ax.tick_params(axis='x', which='minor', length=18, width=0.75)
ax.tick_params(axis='x', which='major', bottom=False)
ax.set_xlim(-0.5, len(datas)-0.5)
ax.set_ylim(ax.get_ylim())
# for x in np.arange(0.5, len(datas)-0.5, 1):
#     plt.plot([x, x], ax.get_ylim(), color='black', linewidth=0.75)
# for dir in ['top','bottom','left','right']:
#     ax.spines[dir].set_linewidth(1)

# plt.yticks(y_ticks, [f'$2^{i}$' for i in range(-3, len(y_ticks)-3)], fontsize=15)
plt.xlabel(None)
plt.ylabel('Workload distribution among TBs (normalized to average)')
plt.legend(loc='best', fontsize=12)


plt.savefig('variance.png', dpi=640, bbox_inches='tight')
plt.savefig('variance.eps', format='eps', bbox_inches='tight')