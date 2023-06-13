import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

if __name__ == "__main__":

    # define default informations
    datas       = ['YouTube', 'IMDB', 'BookCrossing', 'stackoverflow']
    algos       = ['SoTA', 'cuMBE']
    runtimes    = [[8.3, 120, 863, 892],
                   [2.2189, 31.3568, 690, 349.955]]
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