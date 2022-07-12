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
    ''' Plot Fig4. '''

    # Load general data.
    img_ini_0 = mpimg.imread('../image/gray.png')
    img_ini_1 = mpimg.imread('../image/red.png')


    ''' Plot Fig4_1. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(3.8, 3.8 / 3 * 2))
    plt.axis('equal')
    plt.xlim((2.1, 2.1 + 12))
    plt.ylim((-5.8, -5.8 + 8))
    plt.xticks(np.arange(2.1, 2.1 + 12 + 0.1, 3), np.arange(0, 13, 3), family='Arial', fontsize=14)
    plt.yticks(np.arange(-5.8, -5.8 + 8 + 0.1, 2), np.arange(0, 9, 2), family='Arial', fontsize=14)
    plt.subplots_adjust(wspace=0.25, hspace=0.25, left=0.08, bottom=0.11, top=0.96, right=0.93)

    # Load data.
    data = np.load('data/Fig4_1.npz')
    veh_l_1, veh_l_2, veh_w_1, veh_w_2 = 5.85, 4.75, 1.75, 1.8

    # Plot vehicle information.
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.0114 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x1'][-1], data['traj_Re_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.0114 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x2'][-1], data['traj_Re_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['traj_S3_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.0114 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S3_x1'][-1], data['traj_S3_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['traj_S3_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.0114 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S3_x2'][-1], data['traj_S3_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['traj_Re_x1'], data['traj_Re_y1'], color='gray', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_Re_x2'], data['traj_Re_y2'], color='gray', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S3_x1'], data['traj_S3_y1'], color='#FF5050', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S3_x2'], data['traj_S3_y2'], color='#FF5050', linestyle='--', linewidth=1.3, alpha=0.5)

    # Show.
    plt.show()
    # plt.savefig('Fig4_1.png', dpi=600)
    plt.close()


    ''' Plot Fig4_2. '''
    # Basic setup.
    fig_size = (3.8, 3.8 / 3 * 2)
    fig_lim = (12, 8)
    fig, ax = plt.subplots(figsize=fig_size)
    plt.axis('equal')
    plt.xlim((25.5, 25.5 + fig_lim[0]))
    plt.ylim((-6.6, -6.6 + fig_lim[1]))
    plt.xticks(np.arange(25.5, 25.5 + fig_lim[0] + 0.1, 3), np.arange(0, 13, 3), family='Arial', fontsize=14)
    plt.yticks(np.arange(-6.6, -6.6 + fig_lim[1] + 0.1, 2), np.arange(0, 9, 2), family='Arial', fontsize=14)
    plt.subplots_adjust(wspace=0.25, hspace=0.25, left=0.08, bottom=0.11, top=0.96, right=0.93)

    # Load data.
    data = np.load('data/Fig4_2.npz')
    veh_l_1, veh_l_2, veh_w_1, veh_w_2 = 4.53, 4.51, 1.705, 1.725

    # Plot vehicle information.
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.010857 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x1'][-1], data['traj_Re_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.010857 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x2'][-1], data['traj_Re_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['traj_S3_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.010857 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S3_x1'][-1], data['traj_S3_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['traj_S3_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.010857 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S3_x2'][-1], data['traj_S3_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['traj_Re_x1'], data['traj_Re_y1'], color='gray', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_Re_x2'], data['traj_Re_y2'], color='gray', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S3_x1'], data['traj_S3_y1'], color='#FF5050', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S3_x2'], data['traj_S3_y2'], color='#FF5050', linestyle='--', linewidth=1.3, alpha=0.5)

    # Show.
    plt.show()
    # plt.savefig('Fig4_2.png', dpi=600)
    plt.close()

    ''' Plot Fig 4.2.1. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(3, 1.8))
    font1 = {'family': 'Arial', 'size': 14}
    plt.xlabel("Time [ms]", font1, labelpad=-0.6)
    plt.ylabel('Velocity [m/s]', font1, labelpad=3)
    plt.xticks(np.arange(0, 47, 10), np.arange(0, 47, 10) * 10, family='Arial', fontsize=14)
    plt.yticks([15, 20, 25, 30], family='Arial', fontsize=14)
    plt.xlim([-4, 47])
    plt.ylim([12, 33])
    plt.subplots_adjust(left=0.22, wspace=0.25, hspace=0.25, bottom=0.25, top=0.96, right=0.97)

    # Plot Fig4b_1.
    plt.plot(data['traj_Re_V2'][60:], color='lightgray', linestyle='dashed', linewidth=2, zorder=10)
    plt.plot(data['traj_S3_V2'][60:], color='#FF5050', linestyle='dashed', linewidth=2, zorder=7)
    plt.plot(data['traj_Re_V1'][60:], color='lightgray', linestyle='-.', linewidth=2, zorder=5)
    plt.plot(data['traj_S3_V1'][60:], color='#FF5050', linestyle='-.', linewidth=2, zorder=2)

    # Show.
    plt.show()
    # plt.savefig('Fig4_2_1.png', dpi=600)
    plt.close()


    ''' Plot Fig4_3. '''
    # Basic setup.
    fig_size = (3.8, 3.8 / 3 * 2)
    fig_lim = (12, 8)
    fig, ax = plt.subplots(figsize=fig_size)
    plt.axis('equal')
    plt.xlim((-3.3, -3.3 + fig_lim[0]))
    plt.ylim((-7.4, -7.4 + fig_lim[1]))
    plt.xticks(np.arange(-3.3, -3.3 + fig_lim[0] + 0.1, 3), np.arange(0, 13, 3), family='Arial', fontsize=14)
    plt.yticks(np.arange(-7.4, -7.4 + fig_lim[1] + 0.1, 2), np.arange(0, 9, 2), family='Arial', fontsize=14)
    plt.subplots_adjust(wspace=0.25, hspace=0.25, left=0.08, bottom=0.11, top=0.96, right=0.93)

    # Load data.
    data = np.load('data/Fig4_3.npz')
    veh_l_1, veh_l_2, veh_w_1, veh_w_2 = 4.616, 4.416, 1.783, 1.718

    # Plot vehicle information.
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01149 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x1'][-1], data['traj_Re_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01149 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x2'][-1], data['traj_Re_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['traj_S3_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01149 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S3_x1'][-1], data['traj_S3_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['traj_S3_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01149 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S3_x2'][-1], data['traj_S3_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['traj_Re_x1'], data['traj_Re_y1'], color='gray', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_Re_x2'], data['traj_Re_y2'], color='gray', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S3_x1'], data['traj_S3_y1'], color='#FF5050', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S3_x2'], data['traj_S3_y2'], color='#FF5050', linestyle='--', linewidth=1.3, alpha=0.5)

    # Show.
    plt.show()
    # plt.savefig('Fig4_3.png', dpi=600)
    plt.close()


    ''' Plot Fig4_4. '''
    # Basic setup.
    fig_size = (3.8, 3.8 / 3 * 2)
    fig_lim = (12, 8)
    fig, ax = plt.subplots(figsize=fig_size)
    plt.axis('equal')
    plt.xlim((0.5, 0.5 + fig_lim[0]))
    plt.ylim((-0.2, -0.2 + fig_lim[1]))
    plt.xticks(np.arange(0.5, 0.5 + fig_lim[0] + 0.1, 3), np.arange(0, 13, 3), family='Arial', fontsize=14)
    plt.yticks(np.arange(-0.2, -0.2 + fig_lim[1] + 0.1, 2), np.arange(0, 9, 2), family='Arial', fontsize=14)
    plt.subplots_adjust(wspace=0.25, hspace=0.25, left=0.08, bottom=0.11, top=0.96, right=0.93)

    # Load data.
    data = np.load('data/Fig4_4.npz')
    veh_l_1, veh_l_2, veh_w_1, veh_w_2 = 3.995, 4.07, 1.615, 1.615

    # Plot vehicle information.
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01149 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x1'][-1], data['traj_Re_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01149 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x2'][-1], data['traj_Re_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['traj_S3_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01149 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S3_x1'][-1], data['traj_S3_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['traj_S3_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01149 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S3_x2'][-1], data['traj_S3_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['traj_Re_x1'], data['traj_Re_y1'], color='gray', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_Re_x2'], data['traj_Re_y2'], color='gray', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S3_x1'], data['traj_S3_y1'], color='#FF5050', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S3_x2'], data['traj_S3_y2'], color='#FF5050', linestyle='--', linewidth=1.3, alpha=0.5)

    # Show.
    plt.show()
    # plt.savefig('Fig4_4.png', dpi=600)
    plt.close()


    ''' Plot Fig4_5. '''
    # Basic setup.
    fig_size = (3.8, 3.8 / 3 * 2)
    fig_lim = (12, 8)
    fig, ax = plt.subplots(figsize=fig_size)
    plt.axis('equal')
    plt.xlim((14.6, 14.6 + fig_lim[0]))
    plt.ylim((-5.05, -5.05 + fig_lim[1]))
    plt.xticks(np.arange(14.6, 14.6 + fig_lim[0] + 0.1, 3), np.arange(0, 13, 3), family='Arial', fontsize=14)
    plt.yticks(np.arange(-5.05, -5.05 + fig_lim[1] + 0.1, 2), np.arange(0, 9, 2), family='Arial', fontsize=14)
    plt.subplots_adjust(wspace=0.25, hspace=0.25, left=0.08, bottom=0.11, top=0.96, right=0.93)

    # Load data.
    data = np.load('data/Fig4_5.npz')
    veh_l_1, veh_l_2, veh_w_1, veh_w_2 = 4.95, 4.97, 1.78, 1.69

    # Plot vehicle information.
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01149 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x1'][-1], data['traj_Re_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01149 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x2'][-1], data['traj_Re_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['traj_S3_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01149 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S3_x1'][-1], data['traj_S3_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['traj_S3_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01149 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S3_x2'][-1], data['traj_S3_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['traj_Re_x1'], data['traj_Re_y1'], color='gray', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_Re_x2'], data['traj_Re_y2'], color='gray', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S3_x1'], data['traj_S3_y1'], color='#FF5050', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S3_x2'], data['traj_S3_y2'], color='#FF5050', linestyle='--', linewidth=1.3, alpha=0.5)

    # Show.
    plt.show()
    # plt.savefig('Fig4_5.png', dpi=600)
    plt.close()


    ''' Plot Fig4_6. '''
    # Basic setup.
    fig_size = (3.8, 3.8 / 3 * 2)
    fig_lim = (12, 8)
    fig, ax = plt.subplots(figsize=fig_size)
    plt.axis('equal')
    plt.xlim((15.7, 15.7 + fig_lim[0]))
    plt.ylim((-1.35, -1.35 + fig_lim[1]))
    plt.xticks(np.arange(15.7, 15.7 + fig_lim[0] + 0.1, 3), np.arange(0, 13, 3), family='Arial', fontsize=14)
    plt.yticks(np.arange(-1.35, -1.35 + fig_lim[1] + 0.1, 2), np.arange(0, 9, 2), family='Arial', fontsize=14)
    plt.subplots_adjust(wspace=0.25, hspace=0.25, left=0.08, bottom=0.11, top=0.96, right=0.93)

    # Load data.
    data = np.load('data/Fig4_6.npz')
    veh_l_1, veh_l_2, veh_w_1, veh_w_2 = 4.525, 3.588, 1.725, 1.563

    # Plot vehicle information.
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01104 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x1'][-1], data['traj_Re_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01104 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x2'][-1], data['traj_Re_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['traj_S3_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01104 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S3_x1'][-1], data['traj_S3_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['traj_S3_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01104 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S3_x2'][-1], data['traj_S3_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['traj_Re_x1'], data['traj_Re_y1'], color='gray', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_Re_x2'], data['traj_Re_y2'], color='gray', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S3_x1'], data['traj_S3_y1'], color='#FF5050', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S3_x2'], data['traj_S3_y2'], color='#FF5050', linestyle='--', linewidth=1.3, alpha=0.5)

    # Show.
    plt.show()
    # plt.savefig('Fig4_6.png', dpi=600)
    plt.close()


    ''' Plot Fig4_7. '''
    # Basic setup.
    fig_size = (3.8, 3.8 / 3 * 2)
    fig_lim = (12, 8)
    fig, ax = plt.subplots(figsize=fig_size)
    plt.axis('equal')
    plt.xlim((-10.4, -10.4 + fig_lim[0]))
    plt.ylim((-6.6, -6.6 + fig_lim[1]))
    plt.xticks(np.arange(-10.4, -10.4 + fig_lim[0] + 0.1, 3), np.arange(0, 13, 3), family='Arial', fontsize=14)
    plt.yticks(np.arange(-6.6, -6.6 + fig_lim[1] + 0.1, 2), np.arange(0, 9, 2), family='Arial', fontsize=14)
    plt.subplots_adjust(wspace=0.25, hspace=0.25, left=0.08, bottom=0.11, top=0.96, right=0.93)

    # Load data.
    data = np.load('data/Fig4_7.npz')
    veh_l_1, veh_l_2, veh_w_1, veh_w_2 = 3.866, 3.395, 1.622, 1.615

    # Plot vehicle information.
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01118 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x1'][-1], data['traj_Re_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01118 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x2'][-1], data['traj_Re_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['traj_S3_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01118 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S3_x1'][-1], data['traj_S3_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['traj_S3_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01118 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S3_x2'][-1], data['traj_S3_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['traj_Re_x1'], data['traj_Re_y1'], color='gray', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_Re_x2'], data['traj_Re_y2'], color='gray', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S3_x1'], data['traj_S3_y1'], color='#FF5050', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S3_x2'], data['traj_S3_y2'], color='#FF5050', linestyle='--', linewidth=1.3, alpha=0.5)

    # Show.
    plt.show()
    # plt.savefig('Fig4_7.png', dpi=600)
    plt.close()

    ''' Plot Fig 4.7.1. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(3.3 / 1.1, 2.0 / 1.1))
    font1 = {'family': 'Arial', 'size': 15}
    plt.xlabel("Time [ms]", font1, labelpad=-0.3)
    plt.ylabel('Veh I\'s wheel \nangle [°]', font1, labelpad=1)
    plt.xticks(np.arange(0, 55, 25), np.arange(0, 55, 25) * 10, family='Arial', fontsize=14)
    plt.yticks([2, 4, 6, 8, 10], family='Arial', fontsize=14)
    plt.xlim([-2, 52])
    plt.ylim([1.5, 10.5])
    plt.subplots_adjust(left=0.255, wspace=0.25, hspace=0.25, bottom=0.26, top=0.96, right=0.97)

    # Plot Fig4b_1.
    plt.plot(np.rad2deg(data['traj_Re_W1'][60:]), color='lightgray', linestyle='dashed', linewidth=2, zorder=5)
    plt.plot(np.rad2deg(data['traj_S3_W1'][60:]), color='#FF5050', linestyle='dashed', linewidth=2, zorder=2)

    # Show.
    plt.show()
    # plt.savefig('Fig4_7_1.png', dpi=600)
    plt.close()


    ''' Plot Fig4_8. '''
    # Basic setup.
    fig_size = (3.8, 3.8 / 3 * 2)
    fig_lim = (12, 8)
    fig, ax = plt.subplots(figsize=fig_size)
    plt.axis('equal')
    plt.xlim((11, 11 + fig_lim[0]))
    plt.ylim((-7.6, -7.6 + fig_lim[1]))
    fig_size = (3.8, 3.8 / 3 * 2)
    plt.xticks(np.arange(11, 11 + fig_lim[0] + 0.1, 3), np.arange(0, 13, 3), family='Arial', fontsize=14)
    plt.yticks(np.arange(-7.6, -7.6 + fig_lim[1] + 0.1, 2), np.arange(0, 9, 2), family='Arial', fontsize=14)
    plt.subplots_adjust(wspace=0.25, hspace=0.25, left=0.08, bottom=0.11, top=0.96, right=0.93)

    # Load data.
    data = np.load('data/Fig4_8.npz')
    veh_l_1, veh_l_2, veh_w_1, veh_w_2 = 4.616, 4.416, 1.783, 1.718

    # Plot vehicle information.
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01104 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x1'][-1], data['traj_Re_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_0, np.rad2deg(data['traj_Re_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01104 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_Re_x2'][-1], data['traj_Re_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['traj_S3_t1'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.01104 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S3_x1'][-1], data['traj_S3_y1'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['traj_S3_t2'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.01104 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['traj_S3_x2'][-1], data['traj_S3_y2'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['traj_Re_x1'], data['traj_Re_y1'], color='gray', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_Re_x2'], data['traj_Re_y2'], color='gray', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S3_x1'], data['traj_S3_y1'], color='#FF5050', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(data['traj_S3_x2'], data['traj_S3_y2'], color='#FF5050', linestyle='--', linewidth=1.3, alpha=0.5)

    # Show.
    plt.show()
    # plt.savefig('Fig4_8.png', dpi=600)
    plt.close()

    ''' Plot Fig 4.8.1. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(3.3 / 1.1, 2.0 / 1.1))
    font1 = {'family': 'Arial', 'size': 15}
    plt.xlabel("Time [ms]", font1, labelpad=-0.3)
    plt.ylabel('Veh I\'s wheel \nangle [°]', font1, labelpad=5)
    plt.xticks(np.arange(0, 65, 20), np.arange(0, 65, 20) * 10, family='Arial', fontsize=14)
    plt.yticks([0, 2, 4, 6], family='Arial', fontsize=14)
    plt.xlim([-2, 62])
    plt.ylim([-1.3, 7])
    plt.subplots_adjust(left=0.255, wspace=0.25, hspace=0.25, bottom=0.26, top=0.96, right=0.97)

    # Plot Fig4b_1.
    plt.plot(np.rad2deg(data['traj_Re_W1'][50:]), color='lightgray', linestyle='dashed', linewidth=2, zorder=5)
    plt.plot(np.rad2deg(data['traj_S3_W1'][50:]), color='#FF5050', linestyle='dashed', linewidth=2, zorder=2)

    # Show.
    plt.show()
    # plt.savefig('Fig4_8_1.png', dpi=600)
    plt.close()


if __name__ == "__main__":
    main()
