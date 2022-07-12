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
    ''' Plot Fig6b. '''

    # Load general data.
    img_ini_1 = mpimg.imread('../../image/blue_.png')
    img_ini_2 = mpimg.imread('../../image/green_.png')
    img_ini_3 = mpimg.imread('../../image/orange_.png')
    img_ini_4 = mpimg.imread('../../image/red_.png')

    # Load parameters.
    color = ['gray', '#3B89F0', '#41B571', '#FFB70A', '#FF5050']
    veh_l_1, veh_w_1 = 4.825, 1.78
    veh_l_2, veh_w_2 = 4.825, 1.78
    veh_l_3, veh_w_3 = 4.825, 1.78
    veh_l_4, veh_w_4 = 4.825, 1.78


    ''' Plot Fig6b_1. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(3.5 / 21 * 45, 3.5))
    plt.axis('equal')
    plt.xlim((-2.3, 42.7))
    plt.ylim((-5, 16))
    plt.xticks([], [], family='Times New Roman', fontsize=16)
    plt.yticks([], [], family='Times New Roman', fontsize=16)
    plt.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95)

    # Load data.
    data = np.load('data/Fig6b_1.npz')

    # Plot road information.
    ax.plot([26, 26], [6.9, 14.3], color='gray', linewidth=1.8, alpha=0.35)
    ax.plot([10.6, 10.6], [6.9, -0.5], color='gray', linewidth=1.8, alpha=0.35)
    ax.plot([26, 60], [7, 7], color='orange', linewidth=1, alpha=0.5)
    ax.plot([26, 60], [6.8, 6.8], color='orange', linewidth=1, alpha=0.5)
    ax.plot([26, 60], [10.6, 10.6], color='gray', linestyle=(0, (10, 8)), linewidth=1.8, alpha=0.35)
    ax.plot([26, 60], [3.2, 3.2], color='gray', linestyle=(0, (10, 8)), linewidth=1.8, alpha=0.35)
    ax.plot([26, 60], [14.3, 14.3], color='gray', linewidth=2, alpha=0.7)
    ax.plot([26, 60], [-0.5, -0.5], color='gray', linewidth=2, alpha=0.7)
    ax.plot([10.6, -10], [7, 7], color='orange', linewidth=1, alpha=0.5)
    ax.plot([10.6, -10], [6.8, 6.8], color='orange', linewidth=1, alpha=0.5)
    ax.plot([10.6, -10], [10.6, 10.6], color='gray', linestyle=(0, (10, 8)), linewidth=1.8, alpha=0.35)
    ax.plot([10.6, -10], [3.2, 3.2], color='gray', linestyle=(0, (10, 8)), linewidth=1.8, alpha=0.35)
    ax.plot([10.6, -10], [14.3, 14.3], color='gray', linewidth=2, alpha=0.7)
    ax.plot([10.6, -10], [-0.5, -0.5], color='gray', linewidth=2, alpha=0.7)
    ax.plot([22.5, 22.5], [-4, -10], color='gray', linewidth=2, alpha=0.7)
    ax.plot([18.3, 18.3], [-4, -10], color='orange', linestyle=(0, (10, 8)), linewidth=1.8, alpha=0.5)
    ax.plot([14.1, 14.1], [-4, -10], color='#a6a6a6', linewidth=2, )
    ax.plot(26 + 3.5 * np.cos(np.deg2rad(180 - np.arange(101) * 0.9)),
            -4 + 3.5 * np.sin(np.deg2rad(180 - np.arange(101) * 0.9)), color='#a6a6a6', linewidth=2, )
    ax.plot(10.6 + 3.5 * np.cos(np.deg2rad(90 - np.arange(101) * 0.9)),
            -4 + 3.5 * np.sin(np.deg2rad(90 - np.arange(101) * 0.9)), color='#a6a6a6', linewidth=2, )
    ax.plot([22.5, 22.5], [17.8, 20], color='gray', linewidth=2, alpha=0.7)
    ax.plot([18.3, 18.3], [17.8, 20], color='orange', linestyle=(0, (10, 8)), linewidth=1.8, alpha=0.5)
    ax.plot([14.1, 36.6], [17.8, 20], color='#a6a6a6', linewidth=2, )
    ax.plot(26 + 3.5 * np.cos(np.deg2rad(180 + np.arange(101) * 0.9)),
            17.8 + 3.5 * np.sin(np.deg2rad(180 + np.arange(101) * 0.9)), color='#a6a6a6', linewidth=2, )
    ax.plot(10.6 + 3.5 * np.cos(np.deg2rad(-90 + np.arange(101) * 0.9)),
            17.8 + 3.5 * np.sin(np.deg2rad(-90 + np.arange(101) * 0.9)), color='#a6a6a6', linewidth=2, )

    # Plot vehicle information.
    img = resize_rotate(img_ini_4, np.rad2deg(data['V1_t']), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.0062 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['V1_x'], data['V1_y']), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_3, np.rad2deg(data['V2_t']), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.0062 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['V2_x'], data['V2_y']), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_2, np.rad2deg(data['V3_t']), veh_l_3, veh_w_3)
    im = OffsetImage(img, zoom=0.0062 * veh_l_3, alpha=1)
    ab = AnnotationBbox(im, xy=(data['V3_x'], data['V3_y']), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['V4_t']), veh_l_4, veh_w_4)
    im = OffsetImage(img, zoom=0.0062 * veh_l_4, alpha=1)
    ab = AnnotationBbox(im, xy=(data['V4_x'], data['V4_y']), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot vehicle velocity.
    plt.arrow(data['V1_x'], data['V1_y'], 6, 0, width=0.4, head_width=1, head_length=1.5, linewidth=0,
              facecolor=color[4], alpha=1.0, zorder=30)
    plt.arrow(data['V2_x'], data['V2_y'], 2.4, 0, width=0.4, head_width=1, head_length=1.5, linewidth=0,
              facecolor=color[3], alpha=1.0, zorder=30)
    plt.arrow(data['V3_x'], data['V3_y'], 0.98995, 0.98995, width=0.4, head_width=1,
              head_length=1.5, linewidth=0, facecolor=color[2], alpha=1.0, zorder=30)
    plt.arrow(data['V4_x'], data['V4_y'], -3.2, 0, width=0.4, head_width=1, head_length=1.5, linewidth=0,
              facecolor=color[1], alpha=1.0, zorder=30)

    # Show.
    plt.show()
    # plt.savefig('Fig6b_1.png', dpi=600)
    plt.close()


    ''' Plot Fig6b_2. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(1.1 / 5.6 * 14, 1.1))
    plt.axis('equal')
    plt.xlim((14, 14 + 14))
    plt.ylim((1.6, 1.6 + 5.6))
    plt.xticks([], [], family='Times New Roman', fontsize=16)
    plt.yticks([], [], family='Times New Roman', fontsize=16)
    plt.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95)

    # Load data.
    data = np.load('data/Fig6b_2.npz')

    # Plot vehicle information.
    img = resize_rotate(img_ini_4, np.rad2deg(data['V1_t'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.0074 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['V1_x'][-1], data['V1_y'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_3, np.rad2deg(data['V2_t'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.0074 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['V2_x'][-1], data['V2_y'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['V1_x'], data['V1_y'], color=color[4], linestyle='--', linewidth=1.1, alpha=0.8)
    plt.plot(data['V2_x'], data['V2_y'], color=color[3], linestyle='--', linewidth=1.1, alpha=0.8)

    # Show.
    plt.show()
    # plt.savefig('Fig6b_2.png', dpi=600)
    plt.close()


    ''' Plot Fig6b_3. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(1.1 / 5.6 * 14, 1.1))
    plt.axis('equal')
    plt.xlim((16 + 0.5, 30 + 0.5))
    plt.ylim((1.4, 1.4 + 1.1))
    plt.xticks([], [], family='Times New Roman', fontsize=16)
    plt.yticks([], [], family='Times New Roman', fontsize=16)
    plt.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95)

    # Load data.
    data = np.load('data/Fig6b_3.npz')

    # Plot vehicle information.
    img = resize_rotate(img_ini_4, np.rad2deg(data['V1_t'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.0074 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['V1_x'][-1], data['V1_y'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_2, np.rad2deg(data['V3_t'][-1]), veh_l_3, veh_w_3)
    im = OffsetImage(img, zoom=0.0074 * veh_l_3, alpha=1)
    ab = AnnotationBbox(im, xy=(data['V3_x'][-1], data['V3_y'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['V1_x'], data['V1_y'], color=color[4], linestyle='--', linewidth=1.1, alpha=0.8)
    plt.plot(data['V3_x'], data['V3_y'], color=color[2], linestyle='--', linewidth=1.1, alpha=0.8)

    # Show.
    plt.show()
    # plt.savefig('Fig6b_3.png', dpi=600)
    plt.close()


    ''' Plot Fig6b_4. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(1.1 / 5.6 * 14, 1.1))
    plt.axis('equal')
    plt.xlim((16 - 0.9, 30 - 0.9))
    plt.ylim((5 - 0.5, 11. - 0.4 - 0.5))
    plt.xticks([], [], family='Times New Roman', fontsize=16)
    plt.yticks([], [], family='Times New Roman', fontsize=16)
    plt.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95)

    # Load data.
    data = np.load('data/Fig6b_4.npz')

    # Plot vehicle information.
    img = resize_rotate(img_ini_4, np.rad2deg(data['V1_t'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.0074 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['V1_x'][-1], data['V1_y'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['V4_t'][-1]), veh_l_4, veh_w_4)
    im = OffsetImage(img, zoom=0.0074 * veh_l_4, alpha=1)
    ab = AnnotationBbox(im, xy=(data['V4_x'][-1], data['V4_y'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['V1_x'], data['V1_y'], color=color[4], linestyle='--', linewidth=1.1, alpha=0.8)
    plt.plot(data['V4_x'], data['V4_y'], color=color[1], linestyle='--', linewidth=1.1, alpha=0.8)

    # Show.
    plt.show()
    # plt.savefig('Fig6b_4.png', dpi=600)
    plt.close()


    ''' Plot Fig6b_5. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(3.5 / 21 * 45, 3.5))
    plt.axis('equal')
    plt.xlim((-2.3, 42.7))
    plt.ylim((-5, 16))
    plt.xticks([], [], family='Times New Roman', fontsize=16)
    plt.yticks([], [], family='Times New Roman', fontsize=16)
    plt.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95)

    # Load data.
    data = np.load('data/Fig6b_5.npz')

    # Plot road information.
    ax.plot([26, 26], [6.9, 14.3], color='gray', linewidth=1.8, alpha=0.35)
    ax.plot([10.6, 10.6], [6.9, -0.5], color='gray', linewidth=1.8, alpha=0.35)
    ax.plot([26, 60], [7, 7], color='orange', linewidth=1, alpha=0.5)
    ax.plot([26, 60], [6.8, 6.8], color='orange', linewidth=1, alpha=0.5)
    ax.plot([26, 60], [10.6, 10.6], color='gray', linestyle=(0, (10, 8)), linewidth=1.8, alpha=0.35)
    ax.plot([26, 60], [3.2, 3.2], color='gray', linestyle=(0, (10, 8)), linewidth=1.8, alpha=0.35)
    ax.plot([26, 60], [14.3, 14.3], color='gray', linewidth=2, alpha=0.7)
    ax.plot([26, 60], [-0.5, -0.5], color='gray', linewidth=2, alpha=0.7)
    ax.plot([10.6, -10], [7, 7], color='orange', linewidth=1, alpha=0.5)
    ax.plot([10.6, -10], [6.8, 6.8], color='orange', linewidth=1, alpha=0.5)
    ax.plot([10.6, -10], [10.6, 10.6], color='gray', linestyle=(0, (10, 8)), linewidth=1.8, alpha=0.35)
    ax.plot([10.6, -10], [3.2, 3.2], color='gray', linestyle=(0, (10, 8)), linewidth=1.8, alpha=0.35)
    ax.plot([10.6, -10], [14.3, 14.3], color='gray', linewidth=2, alpha=0.7)
    ax.plot([10.6, -10], [-0.5, -0.5], color='gray', linewidth=2, alpha=0.7)
    ax.plot([22.5, 22.5], [-4, -10], color='gray', linewidth=2, alpha=0.7)
    ax.plot([18.3, 18.3], [-4, -10], color='orange', linestyle=(0, (10, 8)), linewidth=1.8, alpha=0.5)
    ax.plot([14.1, 14.1], [-4, -10], color='#a6a6a6', linewidth=2, )
    ax.plot(26 + 3.5 * np.cos(np.deg2rad(180 - np.arange(101) * 0.9)),
            -4 + 3.5 * np.sin(np.deg2rad(180 - np.arange(101) * 0.9)), color='#a6a6a6', linewidth=2, )
    ax.plot(10.6 + 3.5 * np.cos(np.deg2rad(90 - np.arange(101) * 0.9)),
            -4 + 3.5 * np.sin(np.deg2rad(90 - np.arange(101) * 0.9)), color='#a6a6a6', linewidth=2, )
    ax.plot([22.5, 22.5], [17.8, 20], color='gray', linewidth=2, alpha=0.7)
    ax.plot([18.3, 18.3], [17.8, 20], color='orange', linestyle=(0, (10, 8)), linewidth=1.8, alpha=0.5)
    ax.plot([14.1, 36.6], [17.8, 20], color='#a6a6a6', linewidth=2, )
    ax.plot(26 + 3.5 * np.cos(np.deg2rad(180 + np.arange(101) * 0.9)),
            17.8 + 3.5 * np.sin(np.deg2rad(180 + np.arange(101) * 0.9)), color='#a6a6a6', linewidth=2, )
    ax.plot(10.6 + 3.5 * np.cos(np.deg2rad(-90 + np.arange(101) * 0.9)),
            17.8 + 3.5 * np.sin(np.deg2rad(-90 + np.arange(101) * 0.9)), color='#a6a6a6', linewidth=2, )

    # Plot vehicle information.
    img = resize_rotate(img_ini_4, np.rad2deg(data['V1_t']), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.0062 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['V1_x'], data['V1_y']), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_3, np.rad2deg(data['V2_t']), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.0062 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['V2_x'], data['V2_y']), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_2, np.rad2deg(data['V3_t']), veh_l_3, veh_w_3)
    im = OffsetImage(img, zoom=0.0062 * veh_l_3, alpha=1)
    ab = AnnotationBbox(im, xy=(data['V3_x'], data['V3_y']), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['V4_t']), veh_l_4, veh_w_4)
    im = OffsetImage(img, zoom=0.0062 * veh_l_4, alpha=1)
    ab = AnnotationBbox(im, xy=(data['V4_x'], data['V4_y']), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot vehicle velocity.
    plt.arrow(data['V1_x'], data['V1_y'], 6, 0, width=0.4, head_width=1, head_length=1.5, linewidth=0,
              facecolor=color[4], alpha=1.0, zorder=30)
    plt.arrow(data['V3_x'], data['V3_y'], 0.98995, 0.98995, width=0.4, head_width=1,
              head_length=1.5, linewidth=0, facecolor=color[2], alpha=1.0, zorder=30)
    plt.arrow(data['V4_x'], data['V4_y'], -3.2, 0, width=0.4, head_width=1, head_length=1.5, linewidth=0,
              facecolor=color[1], alpha=1.0, zorder=30)

    # Show.
    plt.show()
    # plt.savefig('Fig6b_5.png', dpi=600)
    plt.close()


    ''' Plot Fig6b_6. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(1.1 / 5.6 * 14, 1.1))
    plt.axis('equal')
    plt.xlim((14 - 0.33, 14 - 0.33 + 14))
    plt.ylim((1.6, 1.6 + 5.6))
    plt.xticks([], [], family='Times New Roman', fontsize=16)
    plt.yticks([], [], family='Times New Roman', fontsize=16)
    plt.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95)

    # Load data.
    data = np.load('data/Fig6b_6.npz')

    # Plot vehicle information.
    img = resize_rotate(img_ini_4, np.rad2deg(data['V1_t'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.0074 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['V1_x'][-1], data['V1_y'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_3, np.rad2deg(data['V2_t'][-1]), veh_l_2, veh_w_2)
    im = OffsetImage(img, zoom=0.0074 * veh_l_2, alpha=1)
    ab = AnnotationBbox(im, xy=(data['V2_x'][-1], data['V2_y'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['V1_x'], data['V1_y'], color=color[4], linestyle='--', linewidth=1.1, alpha=0.8)
    plt.plot(data['V2_x'], data['V2_y'], color=color[3], linestyle='--', linewidth=1.1, alpha=0.8)

    # Show.
    plt.show()
    # plt.savefig('Fig6b_6.png', dpi=600)
    plt.close()


    ''' Plot Fig6b_7. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(1.1 / 5.6 * 14, 1.1))
    plt.axis('equal')
    plt.xlim((16 + 0.5, 30 + 0.5))
    plt.ylim((1.4, 1.4 + 1.1))
    plt.xticks([], [], family='Times New Roman', fontsize=16)
    plt.yticks([], [], family='Times New Roman', fontsize=16)
    plt.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95)

    # Load data.
    data = np.load('data/Fig6b_7.npz')

    # Plot vehicle information.
    img = resize_rotate(img_ini_4, np.rad2deg(data['V1_t'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.0074 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['V1_x'][-1], data['V1_y'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_2, np.rad2deg(data['V3_t'][-1]), veh_l_3, veh_w_3)
    im = OffsetImage(img, zoom=0.0074 * veh_l_3, alpha=1)
    ab = AnnotationBbox(im, xy=(data['V3_x'][-1], data['V3_y'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['V1_x'], data['V1_y'], color=color[4], linestyle='--', linewidth=1.1, alpha=0.8)
    plt.plot(data['V3_x'], data['V3_y'], color=color[2], linestyle='--', linewidth=1.1, alpha=0.8)

    # Show.
    plt.show()
    # plt.savefig('Fig6b_7.png', dpi=600)
    plt.close()


    ''' Plot Fig6b_8. '''
    # Basic setup.
    fig, ax = plt.subplots(figsize=(1.1 / 5.6 * 14, 1.1))
    plt.axis('equal')
    plt.xlim((16 - 0.9, 30 - 0.9))
    plt.ylim((5 - 0.5, 11. - 0.4 - 0.5))
    plt.xticks([], [], family='Times New Roman', fontsize=16)
    plt.yticks([], [], family='Times New Roman', fontsize=16)
    plt.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95)

    # Load data.
    data = np.load('data/Fig6b_8.npz')

    # Plot vehicle information.
    img = resize_rotate(img_ini_4, np.rad2deg(data['V1_t'][-1]), veh_l_1, veh_w_1)
    im = OffsetImage(img, zoom=0.0074 * veh_l_1, alpha=1)
    ab = AnnotationBbox(im, xy=(data['V1_x'][-1], data['V1_y'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)
    img = resize_rotate(img_ini_1, np.rad2deg(data['V4_t'][-1]), veh_l_4, veh_w_4)
    im = OffsetImage(img, zoom=0.0074 * veh_l_4, alpha=1)
    ab = AnnotationBbox(im, xy=(data['V4_x'][-1], data['V4_y'][-1]), xycoords='data', pad=0, frameon=False)
    ax.add_artist(ab)

    # Plot trajectory information.
    plt.plot(data['V1_x'], data['V1_y'], color=color[4], linestyle='--', linewidth=1.1, alpha=0.8)
    plt.plot(data['V4_x'], data['V4_y'], color=color[1], linestyle='--', linewidth=1.1, alpha=0.8)

    # Show.
    plt.show()
    # plt.savefig('Fig6b_8.png', dpi=600)
    plt.close()


if __name__ == "__main__":
    main()