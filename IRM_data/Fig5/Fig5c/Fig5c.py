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
    ''' Plot Fig5c. '''

    # Load general data.
    img_ini_00 = mpimg.imread('../../image/gray__.png')
    img_ini_0 = mpimg.imread('../../image/gray.png')
    img_ini_2 = mpimg.imread('../../image/green.png')
    img_ini_3 = mpimg.imread('../../image/orange.png')
    img_ini_4 = mpimg.imread('../../image/red.png')

    # Load parameters.
    veh_l_1, veh_l_2, veh_w_1, veh_w_2 = 4.616, 4.416, 1.783, 1.718
    color = ['gray', '#3B89F0', '#41B571', '#FFB70A', '#FF5050']


    ''' Plot Fig5c_1. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    font1 = {'family': 'Arial', 'size': 15}
    plt.xlabel("Activation time before the collision [ms]", font1)
    plt.ylabel('Reduction of OISS [%]', font1, labelpad=-3.5)
    plt.xticks(np.arange(0, 101, 20), np.arange(0, 101, 20) * 10 - 1000, family='Arial', fontsize=15)
    plt.yticks(family='Arial', fontsize=15)
    plt.subplots_adjust(left=0.15, wspace=0.25, hspace=0.25, bottom=0.13, top=0.97, right=0.97)

    # Load data.
    data = np.load('data/Fig5c_1.npz')

    # Plot Fig5c_1.
    plt.plot(np.arange(0, 101, 10), data['Inj_EB'], color='#3B89F0', marker='o', linestyle='dashed', linewidth=1,
             markersize=6)
    plt.plot(np.arange(0, 101, 10), data['Inj_S1'], color='#41B571', marker='v', linestyle='dashed', linewidth=1,
             markersize=6)
    plt.plot(np.arange(0, 101, 10), data['Inj_S2'], color='#FFB70A', marker='^', linestyle='dashed', linewidth=1,
             markersize=6)
    plt.plot(np.arange(0, 101, 10), data['Inj_S3'], color='#FF5050', marker='s', linestyle='dashed', linewidth=1,
             markersize=6, clip_on=False)

    # Show.
    plt.show()
    # plt.savefig('Fig5c_1.png', dpi=600)
    plt.close()


    ''' Plot Fig5c_2. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(2, 2))
    plt.axis('equal')
    plt.xlim((-6, 34))
    plt.ylim((-16, 24))
    plt.xticks([], family='Arial', fontsize=15)
    plt.yticks([], family='Arial', fontsize=15)
    plt.subplots_adjust(left=0.02, wspace=0.25, hspace=0.25, bottom=0.02, top=0.98, right=0.98)

    # Load data.
    data = np.load('data/Fig5c_2.npz')

    # Plot road information.
    plt.plot([17.8, -35], [-5.7, -5.7], color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot([2.33, -35], [-1.9, -1.9], color='gray', linestyle=(0, (10, 8)), linewidth=1, alpha=0.35)
    plt.plot([-2.7346, -35], [1.9, 1.9], color='gray', linestyle=(0, (10, 8)), linewidth=1, alpha=0.35)
    plt.plot([-7.8, -35], [5.55, 5.55], color='orange', linestyle='-', linewidth=0.7, alpha=0.5)
    plt.plot([-7.8, -35], [5.85, 5.85], color='orange', linestyle='-', linewidth=0.7, alpha=0.5)
    plt.plot([-12.869, -35], [9.5, 9.5], color='gray', linestyle=(0, (10, 8)), linewidth=1, alpha=0.35)
    plt.plot([-17.9346, -35], [13.3, 13.3], color='gray', linestyle=(0, (10, 8)), linewidth=1, alpha=0.35)
    plt.plot([-13, -35], [17.1, 17.1], color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    for i in range(1, 12):
        plt.plot([16 - i * 2.6, 11 - i * 2.6], [-5.7 + 1.9 * i, -5.7 + 1.9 * i], color='lightgray', linestyle='-',
                 linewidth=3, alpha=0.45)
    plt.plot([7.4, -7.8], [-5.7, 5.7], color='gray', linestyle='-', linewidth=1.3, alpha=0.7)

    plt.plot([47.8, 90], [-5.7, -5.7], color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot([52.73, 90], [-1.9, -1.9], color='gray', linestyle=(0, (10, 8)), linewidth=1, alpha=0.35)
    plt.plot([47.6654, 90], [1.9, 1.9], color='gray', linestyle=(0, (10, 8)), linewidth=1, alpha=0.35)
    plt.plot([42.6, 90], [5.55, 5.55], color='orange', linestyle='-', linewidth=0.7, alpha=0.5)
    plt.plot([42.6, 90], [5.85, 5.85], color='orange', linestyle='-', linewidth=0.7, alpha=0.5)
    plt.plot([37.5308, 90], [9.5, 9.5], color='gray', linestyle=(0, (10, 8)), linewidth=1, alpha=0.35)
    plt.plot([32.4654, 90], [13.3, 13.3], color='gray', linestyle=(0, (10, 8)), linewidth=1, alpha=0.35)
    plt.plot([17.4, 90], [17.1, 17.1], color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    for i in range(1, 12):
        plt.plot([50 - i * 2.6, 55 - i * 2.6], [-5.7 + 1.9 * i, -5.7 + 1.9 * i], color='lightgray', linestyle='-',
                 linewidth=3, alpha=0.45)
    plt.plot([27.4, 42.6], [17.1, 5.7], color='gray', linestyle='-', linewidth=1.3, alpha=0.7)

    plt.plot([-35, -13], [33, 17.1], color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot([-27.4, -9.399], [33, 20.1], color='gray', linestyle=(0, (10, 8)), linewidth=1, alpha=0.35)
    plt.plot([-19.5, -1.499], [33, 20.1], color='orange', linestyle='-', linewidth=0.7, alpha=0.5)
    plt.plot([-20.1, -2.1], [33, 20.1], color='orange', linestyle='-', linewidth=0.7, alpha=0.5)
    plt.plot([-12.2, 5.8], [33, 20.1], color='gray', linestyle=(0, (10, 8)), linewidth=1, alpha=0.35)
    plt.plot([-4.6, 17.4], [33, 17.1], color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    for i in range(1, 8):
        plt.plot([13.4 - i * 3.8, 17.4 - i * 3.8], [19.5, 16.8], color='lightgray', linestyle='-', linewidth=3,
                 alpha=0.45)
    plt.plot([-17, -1.8], [20.1, 20.1], color='gray', linestyle='-', linewidth=1.3, alpha=0.7)

    plt.plot([18, 60], [-5.7, -35], color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot([29.6, 67.6], [-8.7, -35], color='gray', linestyle=(0, (10, 8)), linewidth=1, alpha=0.35)
    plt.plot([37.5, 75.5], [-8.7, -35], color='orange', linestyle='-', linewidth=0.7, alpha=0.5)
    plt.plot([36.9, 74.9], [-8.7, -35], color='orange', linestyle='-', linewidth=0.7, alpha=0.5)
    plt.plot([44.8, 82.8], [-8.7, -35], color='gray', linestyle=(0, (10, 8)), linewidth=1, alpha=0.35)
    plt.plot([48.4, 90.4], [-5.7, -35], color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    for i in range(1, 8):
        plt.plot([51.44 - i * 3.8, 48.3741 - i * 3.8], [-8.1, -6],
                 color='lightgray', linestyle='-', linewidth=3, alpha=0.45)
    plt.plot([37.3, 52.6], [-8.7, -8.7], color='gray', linestyle='-', linewidth=1.3, alpha=0.7)

    # Plot vehicle information.
    img = resize_rotate(img_ini_00, np.rad2deg(data['traj_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.003 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_x1'][-1], data['traj_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_00, np.rad2deg(data['traj_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.003 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_x2'][-1], data['traj_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['traj_x1'], data['traj_y1'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_x2'], data['traj_y2'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)

    # Show.
    plt.show()
    # plt.savefig('Fig5c_2.png', dpi=600)
    plt.close()


    ''' Plot Fig5c_3. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(3.5 * 1.1, 3.5 / 9 * 6 * 1.1))
    plt.axis('equal')
    plt.xlim((19.4, 19.4 + 12))
    plt.ylim((-1.8, -1.8 + 8))
    plt.xticks(np.arange(19.4, 19.4 + 13, 3), np.arange(0, 13, 3), family='Arial', fontsize=14)
    plt.yticks(np.arange(-1.8, -1.8 + 9, 2), np.arange(0, 9, 2), family='Arial', fontsize=14)
    plt.subplots_adjust(wspace=0.25, hspace=0.25, left=0.11, bottom=0.11, top=0.96, right=0.96)

    # Load data.
    data = np.load('data/Fig5c_3.npz')

    # Plot vehicle information.
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01164 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x1'][-1], data['traj_Re_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01164 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x2'][-1], data['traj_Re_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_2, np.rad2deg(data['traj_S1_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01164 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S1_x1'][-1], data['traj_S1_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_2, np.rad2deg(data['traj_S1_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01164 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S1_x2'][-1], data['traj_S1_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['traj_Re_x1'], data['traj_Re_y1'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_Re_x2'], data['traj_Re_y2'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S1_x1'], data['traj_S1_y1'], color=color[2], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S1_x2'], data['traj_S1_y2'], color=color[2], linestyle='--', linewidth=1.3, alpha=0.5)

    # Show.
    plt.show()
    # plt.savefig('Fig5c_3.png', dpi=600)
    plt.close()


    ''' Plot Fig5c_4. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(3.5 * 1.1, 3.5 / 9 * 6 * 1.1))
    plt.axis('equal')
    plt.xlim((19.4, 19.4 + 12))
    plt.ylim((-1.8, -1.8 + 8))
    plt.xticks(np.arange(19.4, 19.4 + 13, 3), np.arange(0, 13, 3), family='Arial', fontsize=14)
    plt.yticks(np.arange(-1.8, -1.8 + 9, 2), np.arange(0, 9, 2), family='Arial', fontsize=14)
    plt.subplots_adjust(wspace=0.25, hspace=0.25, left=0.11, bottom=0.11, top=0.96, right=0.96)

    # Load data.
    data = np.load('data/Fig5c_4.npz')

    # Plot vehicle information.
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01164 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x1'][-1], data['traj_Re_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01164 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x2'][-1], data['traj_Re_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_3, np.rad2deg(data['traj_S2_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01164 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S2_x1'][-1], data['traj_S2_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_3, np.rad2deg(data['traj_S2_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01164 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S2_x2'][-1], data['traj_S2_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['traj_Re_x1'], data['traj_Re_y1'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_Re_x2'], data['traj_Re_y2'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S2_x1'], data['traj_S2_y1'], color=color[3], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S2_x2'], data['traj_S2_y2'], color=color[3], linestyle='--', linewidth=1.3, alpha=0.5)

    # Show.
    plt.show()
    # plt.savefig('Fig5c_4.png', dpi=600)
    plt.close()


    ''' Plot Fig5c_5. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(3.5 * 1.1, 3.5 / 9 * 6 * 1.1))
    plt.axis('equal')
    plt.xlim((19.4, 19.4 + 12))
    plt.ylim((-1.8, -1.8 + 8))
    plt.xticks(np.arange(19.4, 19.4 + 13, 3), np.arange(0, 13, 3), family='Arial', fontsize=14)
    plt.yticks(np.arange(-1.8, -1.8 + 9, 2), np.arange(0, 9, 2), family='Arial', fontsize=14)
    plt.subplots_adjust(wspace=0.25, hspace=0.25, left=0.11, bottom=0.11, top=0.96, right=0.96)

    # Load data.
    data = np.load('data/Fig5c_5.npz')

    # Plot vehicle information.
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01164 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x1'][-1], data['traj_Re_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01164 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x2'][-1], data['traj_Re_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_4, np.rad2deg(data['traj_S3_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01164 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S3_x1'][-1], data['traj_S3_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_4, np.rad2deg(data['traj_S3_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01164 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S3_x2'][-1], data['traj_S3_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['traj_Re_x1'], data['traj_Re_y1'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_Re_x2'], data['traj_Re_y2'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S3_x1'], data['traj_S3_y1'], color=color[4], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S3_x2'], data['traj_S3_y2'], color=color[4], linestyle='--', linewidth=1.3, alpha=0.5)

    # Show.
    plt.show()
    # plt.savefig('Fig5c_5.png', dpi=600)
    plt.close()


    ''' Plot Fig5c_6. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(5.15, 1.8))
    font1 = {'family': 'Arial', 'size': 14}
    plt.xlabel("Time [ms]", font1, labelpad=-0.6)
    plt.ylabel('Velocity [m/s]', font1, labelpad=3)
    plt.xticks(np.arange(0, 126, 20), np.arange(0, 126, 20) * 10, family='Arial', fontsize=14)
    plt.yticks(family='Arial', fontsize=14)
    plt.xlim([-5, 125])
    plt.subplots_adjust(left=0.11, wspace=0.25, hspace=0.25, bottom=0.25, top=0.96, right=0.99)

    # Load data.
    data = np.load('data/Fig5c_6.npz')

    # Plot dynamics information.
    plt.plot(data['traj_Re_V1'], color='lightgray', linestyle='dashed', linewidth=2, zorder=10)
    plt.plot(data['traj_S1_V1'], color='#41B571', linestyle='dashed', linewidth=2, zorder=9)
    plt.plot(data['traj_S2_V1'], color='#FFB70A', linestyle='dashed', linewidth=2, zorder=8)
    plt.plot(data['traj_S3_V1'], color='#FF5050', linestyle='dashed', linewidth=2, zorder=7)
    plt.plot(data['traj_Re_V2'], color='lightgray', linestyle='-.', linewidth=2, zorder=5)
    plt.plot(data['traj_S1_V2'], color='#41B571', linestyle='-.', linewidth=2, zorder=3)
    plt.plot(data['traj_S2_V2'], color='#FFB70A', linestyle='-.', linewidth=2, zorder=4)
    plt.plot(data['traj_S3_V2'], color='#FF5050', linestyle='-.', linewidth=2, zorder=2)

    # Show.
    plt.show()
    # plt.savefig('Fig5c_6.png', dpi=600)
    plt.close()


    ''' Plot Fig5c_7. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(5.15, 1.8))
    font1 = {'family': 'Arial', 'size': 14}
    plt.xlabel("Time [ms]", font1, labelpad=-0.6)
    plt.ylabel('Yaw rate [deg/s]', font1, labelpad=1)
    plt.xticks(np.arange(0, 126, 20), np.arange(0, 126, 20) * 10, family='Arial', fontsize=14)
    plt.yticks(family='Arial', fontsize=14)
    plt.xlim([-5, 125])
    plt.subplots_adjust(left=0.13, wspace=0.25, hspace=0.25, bottom=0.25, top=0.96, right=0.99)

    # Load data.
    data = np.load('data/Fig5c_7.npz')

    # Plot dynamics information.
    plt.plot(np.rad2deg(data['traj_Re_W1']), color='lightgray', linestyle='dashed', linewidth=2, zorder=10)
    plt.plot(np.rad2deg(data['traj_S1_W1']), color='#41B571', linestyle='dashed', linewidth=2, zorder=9)
    plt.plot(np.rad2deg(data['traj_S2_W1']), color='#FFB70A', linestyle='dashed', linewidth=2, zorder=8)
    plt.plot(np.rad2deg(data['traj_S3_W1']), color='#FF5050', linestyle='dashed', linewidth=2, zorder=7)
    plt.plot(np.rad2deg(data['traj_Re_W2']), color='lightgray', linestyle='-.', linewidth=2, zorder=5)
    plt.plot(np.rad2deg(data['traj_S1_W2']), color='#41B571', linestyle='-.', linewidth=2, zorder=3)
    plt.plot(np.rad2deg(data['traj_S2_W2']), color='#FFB70A', linestyle='-.', linewidth=2, zorder=4)
    plt.plot(np.rad2deg(data['traj_S3_W2']), color='#FF5050', linestyle='-.', linewidth=2, zorder=2)

    # Show.
    plt.show()
    # plt.savefig('Fig5c_7.png', dpi=600)
    plt.close()


if __name__ == "__main__":
    main()
