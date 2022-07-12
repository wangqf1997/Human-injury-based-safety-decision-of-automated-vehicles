# -*- coding: utf-8 -*-
'''
-------------------------------------------------------------------------------------------------
This code accompanies the paper titled "Human injury-based safety decision of automated vehicles"
Author: Qingfan Wang, Qing Zhou, Miao Lin, Bingbing Nie
Corresponding author: Bingbing Nie (nbb@tsinghua.edu.cn)
-------------------------------------------------------------------------------------------------
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def main():
    ''' Plot Fig5a. '''

    # Basic setup.
    fig, ax = plt.subplots(figsize=(16, 4))

    # Load data.
    Inj_EB = np.load('data/Fig5a.npz')['Inj_EB'].tolist()
    Inj_S1 = np.load('data/Fig5a.npz')['Inj_S1'].tolist()
    Inj_S2 = np.load('data/Fig5a.npz')['Inj_S2'].tolist()
    Inj_S3 = np.load('data/Fig5a.npz')['Inj_S3'].tolist()

    # Define position.
    pos1, pos2, pos3, pos4 = [], [], [], []
    for i in range(11):
        pos1.append(i * 8)
        pos2.append(i * 8 + 1.2)
        pos3.append(i * 8 + 2.4)
        pos4.append(i * 8 + 3.6)

    # Plot injury box figure.
    plt.boxplot(Inj_EB, positions=pos1, whis=0, boxprops={'color': 'black', 'facecolor': '#89C4EF'},
                medianprops={'color': 'red', 'linewidth': 1.3}, showfliers=False, patch_artist=True, widths=1)
    plt.boxplot(Inj_S1, positions=pos2, whis=0, boxprops={'color': 'black', 'facecolor': '#B9EBC7'},
                medianprops={'color': 'red', 'linewidth': 1.3}, showfliers=False, patch_artist=True, widths=1)
    plt.boxplot(Inj_S2, positions=pos3, whis=0, boxprops={'color': 'black', 'facecolor': '#FFBD86'},
                medianprops={'color': 'red', 'linewidth': 1.3}, showfliers=False, patch_artist=True, widths=1)
    plt.boxplot(Inj_S3, positions=pos4, whis=0, boxprops={'color': 'black', 'facecolor': '#FDB7B8'},
                medianprops={'color': 'red', 'linewidth': 1.3}, showfliers=False, patch_artist=True, widths=1)

    # Plot injury line figure.
    data1_median, data2_median, data3_median, data4_median = [], [], [], []
    for i in range(11):
        data1_median.append(np.median(Inj_EB[i]))
        data2_median.append(np.median(Inj_S1[i]))
        data3_median.append(np.median(Inj_S2[i]))
        data4_median.append(np.median(Inj_S3[i]))
    plt.plot(pos1, data1_median, color='#3B89F0', linestyle='dashed', linewidth=2, label='decision', markersize=6)
    plt.plot(pos2, data2_median, color='#41B571', linestyle='dashed', linewidth=2, label='decision', markersize=6)
    plt.plot(pos3, data3_median, color='#FFB70A', linestyle='dashed', linewidth=2, label='slow', markersize=6)
    plt.plot(pos4, data4_median, color='#FF5050', linestyle='dashed', linewidth=2, label='slow', markersize=6)

    # Plot shadow.
    for i in range(11):
        ax.add_patch(plt.Rectangle((-1 + i * 8, -105), 5.8, 220, facecolor='silver', alpha=0.2, ))

    # Figure setup.
    plt.xlim([-3, 87])
    plt.ylim([-3, 103])
    plt.xticks(np.arange(11) * 8 + 1.8, -1000 + np.arange(11) * 100, family='Arial', fontsize=15)
    plt.yticks(family='Arial', fontsize=15)
    plt.grid(axis='y', linestyle="--", alpha=0.8)
    font1 = {'family': 'Arial', 'size': 15}
    plt.xlabel("Activation time before the collision [ms]", font1, labelpad=3.5)
    plt.ylabel('Reduction of OISS [%]', font1, labelpad=-1.5)
    plt.subplots_adjust(left=0.046, wspace=0.25, hspace=0.25, bottom=0.155, top=0.96, right=0.98)

    # Show.
    plt.show()
    # plt.savefig('Fig5a.png', dpi=600)
    plt.close()


if __name__ == "__main__":
    main()