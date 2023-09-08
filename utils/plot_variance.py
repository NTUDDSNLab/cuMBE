import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
import matplotlib.lines as plt_lines
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

# Adjust the widths of a seaborn-generated boxplot.
def adjust_boxplots(g, width_fac=1, shift_dist=0):

    # Iterate through Axes instances
    for ax in g.axes:
        # Iterate through axes artists
        for c in ax.get_children():

            # Determine the type of the artist
            c_type = 'Patch'   if isinstance(c, plt_patches.PathPatch) \
                else 'Line2D'  if isinstance(c, plt_lines.Line2D) \
                else 'Other'

            # Search for PathPatches and Line2Ds
            if c_type != 'Other':

                # Get current width of box:
                p = c.get_path()
                verts = p.vertices[:-1] if c_type == 'Patch' else p.vertices
                xmin  = np.min(verts[:, 0])
                xmax  = np.max(verts[:, 0])
                xmid  = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # Patch : set new width of box
                # Line2D: set new width of maximum & minimum & median lines
                shift_x  = shift_dist if xmid - round(xmid) > 0 else -shift_dist
                xmin_new = xmid - width_fac * xhalf + shift_x
                xmax_new = xmid + width_fac * xhalf + shift_x
                verts[verts[:, 0] == xmin, 0] = xmin_new
                verts[verts[:, 0] == xmax, 0] = xmax_new

# Create a figure and axis for plotting
fig, ax = plt.subplots(figsize=(22.5, 6))

# Initialize lists to store data standard deviations
data_stdevs_WS   = []
data_stdevs_noWS = []

# Create an empty DataFrame for data storage
df = pd.DataFrame(columns=['dataset', 'clock', 'work_stealing'])

# Iterate through each dataset
for data in datas:

    # Open and read the result file for work stealing
    ori_br = open("result/{}_cuMBE_variance".format(data), "r")
    lines = ori_br.readlines()
    del lines[  :5]
    del lines[-2: ]
    res = [eval(i) for i in lines]
    sum = 0
    for num in res:
        sum += num/len(res)

    # Calculate and store the standard deviation
    data_stdevs_WS.append(np.std(res)/sum)

    # Append data to the DataFrame
    for line in res:
        df = df.append({'dataset':'{}'.format(data), 'clock':line/sum, 'work_stealing': 'WS'}, ignore_index=True)

# Repeat the same process for datasets without work stealing
for data in datas:

    ori_br = open("result/{}_noWS_variance".format(data), "r")
    lines = ori_br.readlines()
    del lines[  :5]
    del lines[-2: ]
    res = [eval(i) for i in lines]
    sum = 0
    for num in res:
        sum += num/len(res)

    data_stdevs_noWS.append(np.std(res)/sum)

    for line in res:
        df = df.append({'dataset':'{}'.format(data), 'clock':line/sum, 'work_stealing': 'noWS'}, ignore_index=True)

# Print the standard deviation range
print('[Standard Deviation]')
print('  WS:', min(data_stdevs_WS), '~', max(data_stdevs_WS))
print('noWS:', min(data_stdevs_noWS), '~', max(data_stdevs_noWS))

# Create a boxplot using Seaborn
my_fig = sns.boxplot(x="dataset", y="clock", hue="work_stealing", data=df, showfliers=False, width=0.4, whis=float("inf"))

# Adjust the boxplot widths
adjust_boxplots(fig, shift_dist=0.075)

# Customize x-axis tick labels
xticklabel_objects = my_fig.set_xticklabels(my_fig.get_xticklabels(), fontsize=9)
xticklabel_objects[datas.index('corporate-leadership')].set_fontsize(7.8)

# Add a horizontal grid
plt.grid(axis='y', linestyle='--', color='grey')

# Set y-axis scale to logarithmic (base 2)
plt.yscale('log', base=2)

# Customize y-axis tick labels
y_ticks = [2**i for i in range(math.ceil(np.log2(ax.get_ylim()[0])), math.ceil(np.log2(ax.get_ylim()[1])))]
plt.xticks()
y_labels = [f'{tick:.3f}'.rstrip('0').rstrip('.') if tick != int(tick) else f'{int(tick)}' for tick in y_ticks]
plt.yticks(y_ticks, y_labels)

ax.set_xticks(np.arange(-0.5, len(datas)+0.5, 1), minor=True)
ax.tick_params(axis='x', which='minor', length=18, width=0.75)
ax.tick_params(axis='x', which='major', bottom=False)
ax.set_xlim(-0.5, len(datas)-0.5)

plt.xlabel(None)
plt.ylabel('Workload distribution among TBs (normalized to average)')
plt.legend(loc='best', fontsize=12)

plt.savefig('variance.png', dpi=640, bbox_inches='tight')
plt.savefig('variance.eps', format='eps', bbox_inches='tight')