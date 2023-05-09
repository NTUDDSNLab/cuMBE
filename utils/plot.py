import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

if __name__ == "__main__":

    # define default informations
    datas       = ['arXiv_cond-ma', 'DBpedia_locations', 'cswikisource', 'Marvel', 'YouTube']
    algos       = ['CPU', 'GPU']
    runtimes    = [[2.131176, 392.621765, 0.867390, 14.089269, 4957.191406],
                   [0.309291, 43.111095, 3.936296, 8.527053, 6810.366211]]
    speedups    = [[runtimes[0][x]/runtimes[y][x] for x in range(len(datas))] for y in range(len(algos))]
    figure_name = 'speedup'
    figure_xlim = (0.5, 1.5) if 0 else None
    algo_color = np.linspace((1,1,1), (0,0,0), len(algos)+1, endpoint=False, axis=0)[1:]

    df_speedup = pd.DataFrame({algos[i]:speedups[i] for i in range(len(algos))}, index = datas)

    # plotting with pyplot and pandas module
    fig = plt.figure()
    ax = df_speedup.plot.bar(color=algo_color, edgecolor=['k'])
    # ax.invert_yaxis()
    plt.legend(loc='best')
    plt.xticks(rotation=30)
    plt.xlim(figure_xlim)
    plt.title(figure_name)
    plt.savefig(figure_name+'.png', dpi=320, bbox_inches='tight')
    plt.show()