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
    ''' Plot Fig6a. '''

    # Load general data.
    img_ini_00 = mpimg.imread('../../image/gray__.png')
    img_ini_0 = mpimg.imread('../../image/gray.png')
    img_ini_2 = mpimg.imread('../../image/green.png')
    img_ini_4 = mpimg.imread('../../image/red.png')

    # Load parameters.
    color = ['gray', '#3B89F0', '#41B571', '#FFB70A', '#FF5050']


    ''' Plot Fig6a_1. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(4, 3.3))
    font1 = {'family': 'Arial', 'size': 14}
    plt.xlabel("Activation time before the collision [ms]", font1, labelpad=-0)
    plt.ylabel('Reduction of OISS [%]', font1, labelpad=-3)
    plt.xticks(np.arange(0, 101, 20), np.arange(0, 101, 20) * 10 - 1000, family='Arial', fontsize=14)
    plt.yticks(family='Arial', fontsize=14)
    plt.xlim([-5, 105])
    plt.subplots_adjust(left=0.15, wspace=0.25, hspace=0.25, bottom=0.14, top=0.98, right=0.96)

    # Load data.
    data = np.load('data/Fig6a_1.npz')

    # Plot Fig6a_1.
    plt.plot([-10, 110], [0, 0], color='lightgray', linestyle='dashed')
    plt.plot(np.arange(0, 101, 10), data['Inj_1'], color='#FF5050', marker='s', linestyle='dashed', linewidth=1,
             markersize=6)
    plt.plot(np.arange(0, 101, 10), data['Inj_2'], color='#3B89F0', marker='o', linestyle='dashed', linewidth=1,
             markersize=6)

    # Show.
    plt.show()
    # plt.savefig('Fig6a_1.png', dpi=600)
    plt.close()


    ''' Plot Fig6a_2. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(2, 2))
    plt.axis('equal')
    plt.xlim((-1, -1 + 24))
    plt.ylim((-8.5, -8.5 + 24))
    plt.xticks([], family='Arial', fontsize=14)
    plt.yticks([], family='Arial', fontsize=14)
    plt.subplots_adjust(left=0.02, wspace=0.25, hspace=0.25, bottom=0.02, top=0.98, right=0.98)

    # Load data.
    data = np.load('data/Fig6a_2.npz')
    veh_l_1, veh_l_2, veh_w_1, veh_w_2 = 4.199, 4.649, 1.786, 1.852

    # Plot road information.
    x1 = np.array([-10, 13.5])
    y1 = np.array([-2, -2])
    plt.plot(x1, y1, color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot(x1 - 1, y1 + 3.75, color='orange', linestyle='-', linewidth=0.6, alpha=0.5)
    plt.plot(x1 - 1, y1 + 3.95, color='orange', linestyle='-', linewidth=0.6, alpha=0.5)
    plt.plot(x1, y1 + 7.7, color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot([12.5, 12.5], [-2, 1.85], color='gray', linestyle='-', linewidth=1.3, alpha=0.35)
    x1 = np.array([21.2, 30])
    y1 = np.array([-2, -2])
    plt.plot(x1, y1, color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot(x1 + 1, y1 + 3.75, color='orange', linestyle='-', linewidth=0.6, alpha=0.5)
    plt.plot(x1 + 1, y1 + 3.95, color='orange', linestyle='-', linewidth=0.6, alpha=0.5)
    plt.plot(x1, y1 + 7.7, color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot([22.2, 22.2], [1.85, 5.7], color='gray', linestyle='-', linewidth=1.3, alpha=0.35)
    x2 = np.array([13.5, 13.5])
    y2 = np.array([-10, -2])
    plt.plot(x2, y2, color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot(x2 + 3.75, y2 - 1, color='orange', linestyle='-', linewidth=0.6, alpha=0.5)
    plt.plot(x2 + 3.95, y2 - 1, color='orange', linestyle='-', linewidth=0.6, alpha=0.5)
    plt.plot(x2 + 7.7, y2, color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot([17.35, 21.2], [-3, -3], color='gray', linestyle='-', linewidth=1.3, alpha=0.35)
    x2 = np.array([13.5, 13.5])
    y2 = np.array([5.7, 20])
    plt.plot(x2, y2, color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot(x2 + 3.75, y2 + 1, color='orange', linestyle='-', linewidth=0.6, alpha=0.5)
    plt.plot(x2 + 3.95, y2 + 1, color='orange', linestyle='-', linewidth=0.6, alpha=0.5)
    plt.plot(x2 + 7.7, y2, color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot([13.5, 17.35], [6.7, 6.7], color='gray', linestyle='-', linewidth=1.3, alpha=0.35)

    # Plot vehicle information.
    img = resize_rotate(img_ini_00, np.rad2deg(data['traj_Re_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.0033 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x1'][-1], data['traj_Re_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_00, np.rad2deg(data['traj_Re_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.0033 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x2'][-1], data['traj_Re_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['traj_Re_x1'], data['traj_Re_y1'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_Re_x2'], data['traj_Re_y2'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)

    # Show.
    plt.show()
    # plt.savefig('Fig6a_2.png', dpi=600)
    plt.close()


    ''' Plot Fig6a.3. '''
    # Basic setup.
    fig_size = (3.3, 3.3)
    fig_lim = (8, 8)
    fig, ax = plt.subplots(figsize=fig_size)
    plt.axis('equal')
    plt.xlim((9, 9 + fig_lim[0]))
    plt.ylim((-1, -1 + fig_lim[1]))
    plt.xticks(np.arange(9, 9 + fig_lim[0] + 0.1, 2), np.arange(0, 10, 2), family='Arial', fontsize=14)
    plt.yticks(np.arange(-1, -1 + fig_lim[1] + 0.1, 2), np.arange(0, 10, 2), family='Arial', fontsize=14)
    plt.subplots_adjust(wspace=0.25, hspace=0.25, left=0.08, bottom=0.11, top=0.96, right=0.93)

    # Load data.
    data = np.load('data/Fig6a_3.npz')
    veh_l_1, veh_l_2, veh_w_1, veh_w_2 = 4.199, 4.649, 1.786, 1.852

    # Plot vehicle information.
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01457 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x1'][-1], data['traj_Re_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01457 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x2'][-1], data['traj_Re_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_2, np.rad2deg(data['traj_S1_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01457 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S1_x1'][-1], data['traj_S1_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_2, np.rad2deg(data['traj_S1_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01457 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S1_x2'][-1], data['traj_S1_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['traj_Re_x1'], data['traj_Re_y1'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_Re_x2'], data['traj_Re_y2'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S1_x1'], data['traj_S1_y1'], color=color[2], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S1_x2'], data['traj_S1_y2'], color=color[2], linestyle='--', linewidth=1.3, alpha=0.5)

    # Show.
    plt.show()
    # plt.savefig('Fig6a_3.png', dpi=600)
    plt.close()


    ''' Plot Fig6a_4. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(4, 3.3))
    font1 = {'family': 'Arial', 'size': 14}
    plt.xlabel("Activation time before the collision [ms]", font1, labelpad=-0)
    plt.ylabel('Reduction of OISS [%]', font1, labelpad=-3)
    plt.xticks(np.arange(0, 101, 20), np.arange(0, 101, 20) * 10 - 1000, family='Arial', fontsize=14)
    plt.yticks(family='Arial', fontsize=14)
    plt.xlim([-5, 105])
    plt.subplots_adjust(left=0.15, wspace=0.25, hspace=0.25, bottom=0.14, top=0.98, right=0.96)

    # Load data.
    data = np.load('data/Fig6a_4.npz')

    # Plot Fig6a_4.
    plt.plot([-10, 110], [0, 0], color='lightgray', linestyle='dashed')
    plt.plot(np.arange(0, 101, 10), (data['Inj_1'] + data['Inj_2']) / 2, color='silver', marker='^', linestyle='dashed',
             linewidth=1, markersize=6)
    plt.plot(np.arange(0, 101, 10), data['Inj_1'], color='#FF5050', marker='s', linestyle='dashed', linewidth=1,
             markersize=6)
    plt.plot(np.arange(0, 101, 10), data['Inj_2'], color='#3B89F0', marker='o', linestyle='dashed', linewidth=1,
             markersize=6)

    # Show.
    plt.show()
    # plt.savefig('Fig6a_4.png', dpi=600)
    plt.close()


    ''' Plot Fig6a_5. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(1.2, 2))
    plt.axis('equal')
    plt.xlim((-1, -1 + 28 / 2 * 1.2))
    plt.ylim((-18, -18 + 28))
    plt.xticks([], family='Arial', fontsize=14)
    plt.yticks([], family='Arial', fontsize=14)
    plt.subplots_adjust(left=0.02, wspace=0.25, hspace=0.25, bottom=0.02, top=0.98, right=0.98)

    # Load data.
    data = np.load('data/Fig6a_5.npz')
    veh_l_1, veh_l_2, veh_w_1, veh_w_2 = 4.38, 5.0, 1.73, 1.858

    # Plot road information.
    x1 = np.array([-10, 13.5]) - 9.8
    y1 = np.array([-2, -2])
    plt.plot(x1, y1, color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot(x1 - 1, y1 + 3.75, color='orange', linestyle='-', linewidth=0.6, alpha=0.5)
    plt.plot(x1 - 1, y1 + 3.95, color='orange', linestyle='-', linewidth=0.6, alpha=0.5)
    plt.plot([-19.8, 5.2], y1 + 7.7, color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot([2.7, 2.7], [-2, 1.85], color='gray', linestyle='-', linewidth=1.3, alpha=0.35)
    x1 = np.array([21.2, 40]) - 9.8
    y1 = np.array([-2, -2])
    plt.plot(x1, y1, color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot(x1 + 2.3, y1 + 3.75, color='orange', linestyle='-', linewidth=0.6, alpha=0.5)
    plt.plot(x1 + 2.3, y1 + 3.95, color='orange', linestyle='-', linewidth=0.6, alpha=0.5)
    plt.plot([12.9, 30], y1 + 7.7, color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot([13.7, 13.7], [1.85, 5.7], color='gray', linestyle='-', linewidth=1.3, alpha=0.35)
    x2 = np.array([7, 13.5]) - 9.8
    y2 = np.array([-35, -2])
    plt.plot(x2, y2, color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot(x2 + 3.75, y2 - 1, color='orange', linestyle='-', linewidth=0.6, alpha=0.5)
    plt.plot(x2 + 3.95, y2 - 1, color='orange', linestyle='-', linewidth=0.6, alpha=0.5)
    plt.plot(x2 + 7.7, y2, color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot([7.55, 11.1], [-3, -3], color='gray', linestyle='-', linewidth=1.3, alpha=0.35)
    x2 = np.array([15, 17.8]) - 9.8
    y2 = np.array([5.7, 20])
    plt.plot(x2, y2, color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot(x2 + 3.75, y2 + 1, color='orange', linestyle='-', linewidth=0.6, alpha=0.5)
    plt.plot(x2 + 3.95, y2 + 1, color='orange', linestyle='-', linewidth=0.6, alpha=0.5)
    plt.plot(x2 + 7.7, y2, color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot([5.4, 8.95], [6.7, 6.7], color='gray', linestyle='-', linewidth=1.3, alpha=0.35)

    # Plot vehicle information.
    img = resize_rotate(img_ini_00, np.rad2deg(data['traj_Re_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.0028 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x1'][-1], data['traj_Re_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_00, np.rad2deg(data['traj_Re_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.0028 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x2'][-1], data['traj_Re_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['traj_Re_x1'], data['traj_Re_y1'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_Re_x2'], data['traj_Re_y2'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)

    # Show.
    plt.show()
    # plt.savefig('Fig6a_5.png', dpi=600)
    plt.close()


    ''' Plot Fig6a.6. '''
    # Basic setup.
    fig_size = (3.3, 3.3)
    fig_lim = (8, 8)
    fig, ax = plt.subplots(figsize=fig_size)
    plt.axis('equal')
    plt.xlim((3.9, 3.9 + fig_lim[0]))
    plt.ylim((-6.8, -6.8 + fig_lim[1]))
    plt.xticks(np.arange(3.9, 3.9 + fig_lim[0] + 0.1, 2), np.arange(0, 10, 2), family='Arial', fontsize=14)
    plt.yticks(np.arange(-6.8, -6.8 + fig_lim[1] + 0.1, 2), np.arange(0, 10, 2), family='Arial', fontsize=14)
    plt.subplots_adjust(wspace=0.25, hspace=0.25, left=0.08, bottom=0.11, top=0.96, right=0.93)

    # Load data.
    data = np.load('data/Fig6a_6.npz')
    veh_l_1, veh_l_2, veh_w_1, veh_w_2 = 4.38, 5.0, 1.73, 1.858

    # Plot vehicle information.
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01391 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x1'][-1], data['traj_Re_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01391 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x2'][-1], data['traj_Re_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_4, np.rad2deg(data['traj_S3_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01391 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S3_x1'][-1], data['traj_S3_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_4, np.rad2deg(data['traj_S3_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01391 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S3_x2'][-1], data['traj_S3_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['traj_Re_x1'], data['traj_Re_y1'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_Re_x2'], data['traj_Re_y2'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S3_x1'], data['traj_S3_y1'], color=color[4], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S3_x2'], data['traj_S3_y2'], color=color[4], linestyle='--', linewidth=1.3, alpha=0.5)

    # Show.
    plt.show()
    # plt.savefig('Fig6a_6.png', dpi=600)
    plt.close()


if __name__ == "__main__":
    main()