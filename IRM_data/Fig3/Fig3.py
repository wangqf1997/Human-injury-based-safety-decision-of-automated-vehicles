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
import seaborn as sns
import pandas as pd


def resize_rotate(image, angle, l_, w_):
    ''' resize and rotate the figure. '''

    image = cv2.resize(image, (image.shape[1], int(image.shape[0] / (3370 / 8651) * (w_ / l_))))

    # grab the dimensions of the image and then determine the center.
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix and the sine and cosine.
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image.
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation.
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image.
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))


def main():
    ''' Plot Fig3a. '''

    # Basic setup.
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    plt.axis('equal')
    plt.xlim((-8, 8))
    plt.ylim((-7 / 9 * 8 - 0.2, 7 / 9 * 8 - 0.2))
    plt.xticks([], [], family='Arial', fontsize=16)
    plt.yticks([], [], family='Arial', fontsize=16)
    plt.subplots_adjust(left=0.04, bottom=0.04, top=0.96, right=0.96)

    # four types of collision condition.
    front = [1, 3, 4, 5, 7, 15, 16, 23, 26, 31, 37, 40, 42, 44, 45, 46, 47, 49]
    left = [0, 13, 14, 30, 33, 34, 35, 36, 38]
    right = [2, 8, 9, 11, 12, 17, 18, 21, 22, 24, 27, 28, 29, 39, 41, 43]
    rear = [6, 10, 19, 20, 25, 32, 48]

    # Load data.
    img_ini_0 = mpimg.imread('../image/gray.png')
    img_ini_1 = mpimg.imread('../image/blue.png')
    img_ini_2 = mpimg.imread('../image/green.png')
    img_ini_3 = mpimg.imread('../image/orange.png')
    img_ini_4 = mpimg.imread('../image/red.png')

    img_ini_list = [img_ini_0, img_ini_1, img_ini_2, img_ini_3, img_ini_4]
    data = np.load('data/Fig3a.npz')

    # Plot Fig3a.
    for i in range(50):
        if i in front:
            img_ini = img_ini_list[1]
        elif i in left:
            img_ini = img_ini_list[2]
        elif i in right:
            img_ini = img_ini_list[3]
        elif i in rear:
            img_ini = img_ini_list[4]

        img = resize_rotate(img_ini_0, np.rad2deg(data['traj_t1'][i]), data['l1'][i], data['w1'][i])
        im = OffsetImage(img, zoom=0.01065 * data['l1'][i], alpha=1)
        ab = AnnotationBbox(im, xy=(data['traj_x1'][i], data['traj_y1'][i]), xycoords='data', pad=0, frameon=False)
        ax.add_artist(ab)
        img = resize_rotate(img_ini, np.rad2deg(data['traj_t2'][i]), data['l2'][i], data['w2'][i])
        im = OffsetImage(img, zoom=0.01065 * data['l2'][i], alpha=1)
        ab = AnnotationBbox(im, xy=(data['traj_x2'][i], data['traj_y2'][i]), xycoords='data', pad=0, frameon=False)
        ax.add_artist(ab)

    # Show.
    plt.show()
    # plt.savefig('Fig3a.png', dpi=600)
    plt.close()


    ''' Plot Fig3b. '''

    # Basic setup.
    fig, ax = plt.subplots(figsize=(5, 4))
    plt.xlim((0, 30))
    plt.ylim((0, 20))
    plt.xticks(np.arange(0, 30.1, 5), family='Arial', fontsize=18)
    plt.yticks(np.arange(0, 20.1, 5), family='Arial', fontsize=18)
    plt.xlabel('Velocity [m/s]', family='Arial', fontsize=18)
    plt.ylabel('Delta_$v$ [m/s]', family='Arial', fontsize=18)
    plt.subplots_adjust(left=0.162, bottom=0.16, top=0.96, right=0.962)

    # four types of collision condition.
    front = [1, 2, 3, 4, 9, 13, 14, 15, 21, 24, 29, 31, 37, 40, 42, 44, 45, 46, 47, 49]
    left = [0, 10, 11, 12, 30, 33, 34, 35, 36, 38]
    right = [6, 7, 16, 19, 20, 22, 25, 26, 27, 28, 39, 41, 43]
    rear = [5, 8, 17, 18, 23, 32, 48]

    # Load data.
    data = np.load('data/Fig3b.npz')
    velocity = data['velocity']
    delta_v = data['delta_v']

    # Plot Fig3b.
    for i in range(50):
        if i in front:
            color = '#3B89F0'
            shape = 'o'
        elif i in left:
            color = '#41B571'
            shape = 'v'
        elif i in right:
            color = '#FFB70A'
            shape = '^'
        elif i in rear:
            color = '#FF5050'
            shape = 's'
        plt.plot(velocity[i], delta_v[i], shape, color=color, linestyle='--', linewidth=1, alpha=0.7)

    # Show.
    plt.show()
    # plt.savefig('Fig3b.png', dpi=600)
    plt.close()


    ''' Plot Fig3c. '''

    # Load data.
    data = np.load('data/Fig3b.npz')
    velocity = data['velocity']
    delta_v = data['delta_v']
    collision = [1, 0, 0, 0, 0,  3, 2, 2, 3, 0,
                 1, 1, 1, 0, 0,  0, 2, 3, 3, 2,
                 2, 0, 2, 3, 0,  2, 2, 2, 2, 0,
                 1, 0, 3, 1, 1,  1, 1, 0, 1, 2,
                 0, 2, 0, 2, 0,  0, 0, 0, 3, 0]
    collision = np.array([collision, collision]).reshape(-1)

    # Plot Fig3c.
    dic1 = {'velocity': velocity.transpose().reshape(-1), 'deltaV': delta_v.transpose().reshape(-1),
            'collision': collision}
    df = pd.DataFrame(dic1)
    h = sns.jointplot(data=df, x='velocity', y='deltaV', hue="collision", kind="scatter", xlim=[0, 30], ylim=[0, 20],
                      s=15, palette=['#3B89F0', '#41B571', '#FFB70A', '#FF5050'], legend=False)

    # Figure setup.
    h.ax_joint.set_xlabel('Velocity [m/s]', family='Arial', fontsize=18)
    h.ax_joint.set_ylabel('Delta-v [m/s]', family='Arial', fontsize=18)
    h.ax_joint.set_xticks(range(0, 31, 5))
    h.ax_joint.set_yticks(range(0, 21, 5))
    h.ax_joint.set_xticklabels(np.arange(0, 31, 5), family='Arial', fontsize=18)
    h.ax_joint.set_yticklabels(np.arange(0, 21, 5), family='Arial', fontsize=18)
    h.fig.set_size_inches(5, 4)

    # Show.
    plt.show()
    # plt.savefig('Fig3c.png', dpi=600)
    plt.close()


if __name__ == "__main__":
    main()
