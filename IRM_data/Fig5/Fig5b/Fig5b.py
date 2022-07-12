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
    ''' Plot Fig5b. '''

    # Load general data.
    img_ini_00 = mpimg.imread('../../image/gray__.png')
    img_ini_0 = mpimg.imread('../../image/gray.png')
    img_ini_1 = mpimg.imread('../../image/blue.png')
    img_ini_2 = mpimg.imread('../../image/green.png')
    img_ini_3 = mpimg.imread('../../image/orange.png')

    # Load parameters.
    veh_l_1, veh_l_2, veh_w_1, veh_w_2 = 3.995, 4.07, 1.615, 1.615
    color = ['gray', '#3B89F0', '#41B571', '#FFB70A', '#FF5050']


    ''' Plot Fig5b_1. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    font1 = {'family': 'Arial', 'size': 15}
    plt.xlabel("Activation time before the collision [ms]", font1)
    plt.ylabel('Reduction of OISS [%]', font1, labelpad=-3.5)
    plt.xticks(np.arange(0, 101, 20), np.arange(0, 101, 20) * 10 - 1000, family='Arial', fontsize=15)
    plt.yticks(family='Arial', fontsize=15)
    plt.subplots_adjust(left=0.15, wspace=0.25, hspace=0.25, bottom=0.13, top=0.97, right=0.97)

    # Load data.
    data = np.load('data/Fig5b_1.npz')

    # Plot Fig5b_1.
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
    # plt.savefig('Fig5b_1.png', dpi=600)
    plt.close()


    ''' Plot Fig5b_2. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(2, 2 / 28 * 24))
    plt.axis('equal')
    plt.xlim((-4, 24))
    plt.ylim((-11 + 0.5, 13 + 0.5))
    plt.xticks([], family='Arial', fontsize=15)
    plt.yticks([], family='Arial', fontsize=15)
    plt.subplots_adjust(left=0.02, wspace=0.25, hspace=0.25, bottom=0.02, top=0.98, right=0.98)

    # Load data.
    data = np.load('data/Fig5b_2.npz')

    # Plot road information.
    x = data['road_x']
    y = data['road_y']
    plt.plot(x + 3.5, y - 1, color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot(x[:35] + 3.5, y[:35] - 8.6, color='orange', linestyle='-', linewidth=0.7, alpha=0.5)
    plt.plot(x[:35] + 3.5, y[:35] - 8.6, color='orange', linestyle='-', linewidth=0.7, alpha=0.5)
    plt.plot(x[0:36][::-1] + 3.5, y[:36][::-1] - 4.75, color='gray', linestyle=(0, (10, 8)), linewidth=1, alpha=0.35)
    plt.plot(x[0:33][::-1] + 3.5, y[:33][::-1] - 12.25, color='gray', linestyle=(0, (10, 8)), linewidth=1, alpha=0.35)
    plt.plot(x[:-65] + 3.5, y[:-65] - 16, color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot(x[:-65] + 3.5, -y[:-65] + 10.508, color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot(x[:-68][::-1] - 4, -y[:-68][::-1] + 8.508, color='orange', linestyle=(0, (10, 8)), linewidth=1, alpha=0.5)
    plt.plot(x[:-10] - 0.5, -y[:-10] + 3.508, color='gray', linestyle='-', linewidth=1.3, alpha=0.7)
    plt.plot([x[34] + 3.5, x[30] - 0.3], [y[34] - 8.5, y[30] - 2], color='gray', linestyle='-', linewidth=1.3,
             alpha=0.5)
    plt.plot(x[70:] + 3.5, y[70:] - 8.7, color='orange', linestyle='-', linewidth=0.7, alpha=0.5)
    plt.plot(x[70:] + 3.5, y[70:] - 8.9, color='orange', linestyle='-', linewidth=0.7, alpha=0.5)
    plt.plot(x[70:] + 3.6, y[70:] - 4.95, color='gray', linestyle=(0, (10, 8)), linewidth=1, alpha=0.35)
    plt.plot([x[70] + 3.5, x[70] + 3.6], [y[70] - 8.8, y[70] - 17], color='gray', linestyle='-', linewidth=1.3,
             alpha=0.5)
    plt.plot(x[70:] + 3.5, y[70:] - 12.95, color='gray', linestyle=(0, (10, 8)), linewidth=1, alpha=0.35)

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
    # plt.savefig('Fig5b_2.png', dpi=600)
    plt.close()


    ''' Plot Fig5b_3. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(3.5 * 1.1, 3.5 / 9 * 6 * 1.1))
    plt.axis('equal')
    plt.xlim((5.5, 5.5 + 9))
    plt.ylim((-3.8, -3.8 + 6))
    plt.xticks(np.arange(5.5, 5.5 + 10, 3), np.arange(0, 10, 3), family='Arial', fontsize=14)
    plt.yticks(np.arange(-3.8, -3.8 + 8, 3), np.arange(0, 8, 3), family='Arial', fontsize=14)
    plt.subplots_adjust(wspace=0.25, hspace=0.25, left=0.11, bottom=0.11, top=0.96, right=0.96)

    # Load data.
    data = np.load('data/Fig5b_3.npz')

    # Plot vehicle information.
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.0155 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x1'][-1], data['traj_Re_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.0155 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x2'][-1], data['traj_Re_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['traj_EB_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.0155 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_EB_x1'][-1], data['traj_EB_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['traj_EB_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.0155 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_EB_x2'][-1], data['traj_EB_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['traj_Re_x1'], data['traj_Re_y1'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_Re_x2'], data['traj_Re_y2'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_EB_x1'], data['traj_EB_y1'], color=color[1], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_EB_x2'], data['traj_EB_y2'], color=color[1], linestyle='--', linewidth=1.3, alpha=0.5)

    # Show.
    plt.show()
    # plt.savefig('Fig5b_3.png', dpi=600)
    plt.close()


    ''' Plot Fig5b_4. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(3.5 * 1.1, 3.5 / 9 * 6 * 1.1))
    plt.axis('equal')
    plt.xlim((5.5, 5.5 + 9))
    plt.ylim((-3.8, -3.8 + 6))
    plt.xticks(np.arange(5.5, 5.5 + 10, 3), np.arange(0, 10, 3), family='Arial', fontsize=14)
    plt.yticks(np.arange(-3.8, -3.8 + 8, 3), np.arange(0, 8, 3), family='Arial', fontsize=14)
    plt.subplots_adjust(wspace=0.25, hspace=0.25, left=0.11, bottom=0.11, top=0.96, right=0.96)

    # Load data.
    data = np.load('data/Fig5b_4.npz')

    # Plot vehicle information.
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.0155 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x1'][-1], data['traj_Re_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.0155 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x2'][-1], data['traj_Re_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_2, np.rad2deg(data['traj_S1_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.0155 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S1_x1'][-1], data['traj_S1_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_2, np.rad2deg(data['traj_S1_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.0155 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S1_x2'][-1], data['traj_S1_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['traj_Re_x1'], data['traj_Re_y1'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_Re_x2'], data['traj_Re_y2'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S1_x1'], data['traj_S1_y1'], color=color[2], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S1_x2'], data['traj_S1_y2'], color=color[2], linestyle='--', linewidth=1.3, alpha=0.5)

    # Show.
    plt.show()
    # plt.savefig('Fig5b_4.png', dpi=600)
    plt.close()


    ''' Plot Fig5b_5. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(3.5 * 1.1, 3.5 / 9 * 6 * 1.1))
    plt.axis('equal')
    plt.xlim((5.5, 5.5 + 9))
    plt.ylim((-3.8, -3.8 + 6))
    plt.xticks(np.arange(5.5, 5.5 + 10, 3), np.arange(0, 10, 3), family='Arial', fontsize=14)
    plt.yticks(np.arange(-3.8, -3.8 + 8, 3), np.arange(0, 8, 3), family='Arial', fontsize=14)
    plt.subplots_adjust(wspace=0.25, hspace=0.25, left=0.11, bottom=0.11, top=0.96, right=0.96)

    # Load data.
    data = np.load('data/Fig5b_5.npz')

    # Plot vehicle information.
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.0155 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x1'][-1], data['traj_Re_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.0155 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x2'][-1], data['traj_Re_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_3, np.rad2deg(data['traj_S2_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.0155 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S2_x1'][-1], data['traj_S2_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_3, np.rad2deg(data['traj_S2_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.0155 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S2_x2'][-1], data['traj_S2_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['traj_Re_x1'], data['traj_Re_y1'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_Re_x2'], data['traj_Re_y2'], color=color[0], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S2_x1'], data['traj_S2_y1'], color=color[3], linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S2_x2'], data['traj_S2_y2'], color=color[3], linestyle='--', linewidth=1.3, alpha=0.5)

    # Show.
    plt.show()
    # plt.savefig('Fig5b_5.png', dpi=600)
    plt.close()


    ''' Plot Fig5b_6. '''
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
    data = np.load('data/Fig5b_6.npz')

    # Plot dynamics information.
    plt.plot(data['traj_Re_V1'], color='lightgray', linestyle='dashed', linewidth=2, zorder=10)
    plt.plot(data['traj_EB_V1'], color='#3B89F0', linestyle='dashed', linewidth=2, zorder=9)
    plt.plot(data['traj_S1_V1'], color='#41B571', linestyle='dashed', linewidth=2, zorder=8)
    plt.plot(data['traj_S2_V1'], color='#FFB70A', linestyle='dashed', linewidth=2, zorder=7)
    plt.plot(data['traj_Re_V2'], color='lightgray', linestyle='-.', linewidth=2, zorder=5)
    plt.plot(data['traj_EB_V2'], color='#3B89F0', linestyle='-.', linewidth=2, zorder=4)
    plt.plot(data['traj_S1_V2'], color='#41B571', linestyle='-.', linewidth=2, zorder=3)
    plt.plot(data['traj_S2_V2'], color='#FFB70A', linestyle='-.', linewidth=2, zorder=2)

    # Show.
    plt.show()
    # plt.savefig('Fig5b_6.png', dpi=600)
    plt.close()


    ''' Plot Fig5b_7. '''
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
    data = np.load('data/Fig5b_7.npz')

    # Plot dynamics information.
    plt.plot(np.rad2deg(data['traj_Re_W1']), color='lightgray', linestyle='dashed', linewidth=2, zorder=10)
    plt.plot(np.rad2deg(data['traj_EB_W1']), color='#3B89F0', linestyle='dashed', linewidth=2, zorder=9)
    plt.plot(np.rad2deg(data['traj_S1_W1']), color='#41B571', linestyle='dashed', linewidth=2, zorder=8)
    plt.plot(np.rad2deg(data['traj_S2_W1']), color='#FFB70A', linestyle='dashed', linewidth=2, zorder=7)
    plt.plot(np.rad2deg(data['traj_Re_W2']), color='lightgray', linestyle='-.', linewidth=2, zorder=5)
    plt.plot(np.rad2deg(data['traj_EB_W2']), color='#3B89F0', linestyle='-.', linewidth=2, zorder=4)
    plt.plot(np.rad2deg(data['traj_S1_W2']), color='#41B571', linestyle='-.', linewidth=2, zorder=3)
    plt.plot(np.rad2deg(data['traj_S2_W2']), color='#FFB70A', linestyle='-.', linewidth=2, zorder=2)

    # Show.
    plt.show()
    # plt.savefig('Fig5b_7.png', dpi=600)
    plt.close()


if __name__ == "__main__":
    main()
