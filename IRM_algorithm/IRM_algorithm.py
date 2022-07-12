# -*- coding: utf-8 -*-
'''
-------------------------------------------------------------------------------------------------
This code accompanies the paper titled "Human injury-based safety decision of automated vehicles"
Author: Qingfan Wang, Qing Zhou, Miao Lin, Bingbing Nie
Corresponding author: Bingbing Nie (nbb@tsinghua.edu.cn)
-------------------------------------------------------------------------------------------------
'''


import argparse
import cv2
import xlrd
import torch
import imageio
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from utils.Percept import percept
from utils.Det_crash import det_crash
from utils.Vehicle import Vehicle_S12, Vehicle_S3, deci_S3, deci_EB
from utils.Inj_Pre import RNN
from utils.Con_est import Collision_cond


warnings.filterwarnings('ignore')


__author__ = "Qingfan Wang"


def load_para(file, Num):
    ''' Load the reconstructed information of real-world accidents. '''

    # Load the data file.
    para_data = xlrd.open_workbook(file).sheet_by_index(0)

    # Load and process the vehicle parameters.
    veh_l = [para_data.row_values(Num + 1)[3] / 1000, para_data.row_values(Num + 1 + 51)[3] / 1000]
    veh_w = [para_data.row_values(Num + 1)[4] / 1000, para_data.row_values(Num + 1 + 51)[4] / 1000]
    veh_cgf = [para_data.row_values(Num + 1)[6], para_data.row_values(Num + 1 + 51)[6]]
    veh_cgs = [0.5 * veh_w[0], 0.5 * veh_w[1]]
    veh_m = [para_data.row_values(Num + 1)[2], para_data.row_values(Num + 1 + 51)[2]]
    veh_I = [para_data.row_values(Num + 1)[5], para_data.row_values(Num + 1 + 51)[5]]
    veh_k = [np.sqrt(veh_I[0] / veh_m[0]), np.sqrt(veh_I[1] / veh_m[1])]
    veh_param = (veh_l, veh_w, veh_cgf, veh_cgs, veh_k, veh_m)

    # Load and process the occupant parameters.
    age = [para_data.row_values(Num + 1)[8], para_data.row_values(Num + 1 + 51)[8]]
    sex = [para_data.row_values(Num + 1)[7], para_data.row_values(Num + 1 + 51)[7]]
    belt = [para_data.row_values(Num + 1)[9], para_data.row_values(Num + 1 + 51)[9]]
    airbag = [para_data.row_values(Num + 1)[10], para_data.row_values(Num + 1 + 51)[10]]

    for i in range(2):
        if age[i] < 20:
            age[i] = 0
        elif age[i] < 45:
            age[i] = 1
        elif age[i] < 65:
            age[i] = 2
        else:
            age[i] = 3

    belt[0] = 0 if belt[0] == 'Not in use' else 1
    belt[1] = 0 if belt[1] == 'Not in use' else 1
    sex[0] = 0 if sex[0] == 'Male' else 1
    sex[1] = 0 if sex[1] == 'Male' else 1
    airbag[0] = 1 if airbag[0] == 'Activated' else 0
    airbag[1] = 1 if airbag[1] == 'Activated' else 0

    mass_r = veh_m[0] / veh_m[1]
    if mass_r < 1 / 2:
        mass_r_12 = 0
    elif mass_r < 1 / 1.3:
        mass_r_12 = 1
    elif mass_r < 1.3:
        mass_r_12 = 2
    elif mass_r < 2:
        mass_r_12 = 3
    else:
        mass_r_12 = 4
    mass_r_21 = 4 - mass_r_12
    mass_r = [mass_r_12, mass_r_21]

    occ_param = (age, belt, sex, airbag, mass_r)

    return veh_param, occ_param


def resize_pic(image, angle, l_, w_):
    ''' Resize and rotate the vehicle.png. '''

    # Resize the picture.
    image = cv2.resize(image, (image.shape[1], int(image.shape[0] / (3370 / 8651) * (w_ / l_))))

    # Obtain the dimensions of the image and then determine the center.
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # Obtain the rotation matrix.
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
    image_ = cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))

    return image_


def plot_env(ax, V1_x_seq, V1_y_seq, V1_angle, V2_x_seq, V2_y_seq, V2_angle, veh_l, veh_w, img_list):
    ''' Visualize the simulation environment. '''

    plt.cla()
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlim((-10, 40))
    plt.ylim((-10, 20))
    plt.xticks(np.arange(-10, 40.1, 5), range(0, 50+1, 5), family='Times New Roman', fontsize=16)
    plt.yticks(np.arange(-10, 20.1, 5), range(0, 30+1, 5), family='Times New Roman', fontsize=16)
    plt.subplots_adjust(left=0.1, bottom=0.1, top=0.94, right=0.94, wspace=0.25, hspace=0.25)

    # # Plot the vehicles' position.
    # zoom = ((40) - (-10)) * 0.000135
    # img_1 = resize_pic(img_list[0], np.rad2deg(V1_angle), veh_l[0], veh_w[0])
    # im_1 = OffsetImage(img_1, zoom=zoom * veh_l[0])
    # ab_1 = AnnotationBbox(im_1, xy=(V1_x_seq[-1], V1_y_seq[-1]), xycoords='data', pad=0, frameon=False)
    # ax.add_artist(ab_1)
    # img_2 = resize_pic(img_list[1], np.rad2deg(V2_angle), veh_l[1], veh_w[1])
    # im_2 = OffsetImage(img_2, zoom=zoom * veh_l[1])
    # ab_2 = AnnotationBbox(im_2, xy=(V2_x_seq[-1], V2_y_seq[-1]), xycoords='data', pad=0, frameon=False)
    # ax.add_artist(ab_2)

    from matplotlib import patches
    p_x1 = V1_x_seq[-1] - (veh_l[0] / 2) * np.cos(V1_angle) + (veh_w[0] / 2) * np.sin(V1_angle)
    p_y1 = V1_y_seq[-1] - (veh_l[0] / 2) * np.sin(V1_angle) - (veh_w[0] / 2) * np.cos(V1_angle)
    p_x2 = V2_x_seq[-1] - (veh_l[1] / 2) * np.cos(V2_angle) + (veh_w[1] / 2) * np.sin(V2_angle)
    p_y2 = V2_y_seq[-1] - (veh_l[1] / 2) * np.sin(V2_angle) - (veh_w[1] / 2) * np.cos(V2_angle)
    e1 = patches.Rectangle((p_x1, p_y1), veh_l[0], veh_w[0], angle=np.rad2deg(V1_angle), linewidth=0, fill=True,
                           zorder=2, color='red', alpha=0.5)
    ax.add_patch(e1)
    e2 = patches.Rectangle((p_x2, p_y2), veh_l[1], veh_w[1], angle=np.rad2deg(V2_angle), linewidth=0, fill=True,
                           zorder=2, color='blue', alpha=0.5)
    ax.add_patch(e2)


    # Plot the vehicles' trajectories.
    plt.plot(V1_x_seq, V1_y_seq, color='red', linestyle='--', linewidth=1.3, alpha=0.5)
    plt.plot(V2_x_seq, V2_y_seq, color='blue', linestyle='--', linewidth=1.3, alpha=0.5)


def main():
    ''' Make human injury-based safety decisions using the injury risk mitigation (IRM) algorithm. '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--case_num', type=int, default=1, help='Simulation case number (1-5)')
    parser.add_argument('--t_act', type=int, default=500,
                        help='Activation time of IRM algorithm (100ms~1000ms before the collision)')
    parser.add_argument('--Level', type=str, default='S1', help='Level of the IRM algorithm: EB, S1, S2, S3')
    parser.add_argument('--Ego_V', type=int, default=1, help='Choose one vehicle as the ego vehicle: 1 or 2')
    parser.add_argument('--profile_inf', type=str, default='para\Record_Information_example.xlsx',
                        help='File: information of the reconstructed accidents')
    parser.add_argument('--no_visualize', action='store_false', help='simulation visualization')
    parser.add_argument('--save_gif', action='store_true', help='save simulation visualization')
    opt = parser.parse_args()

    Deci_set = ['straight_cons', 'straight_dec-all', 'straight_dec-half', 'straight_acc-half', 'straight_acc-all',
                'left-all_dec-all', 'right-all_dec-all', 'left-all_acc-all', 'right-all_acc-all',
                'left-half_dec-all', 'left-half_dec-half', 'left-all_dec-half', 'left-half_cons',
                'left-all_cons', 'left-half_acc-half', 'left-all_acc-half', 'left-half_acc-all',
                'right-half_dec-all', 'right-half_dec-half', 'right-all_dec-half', 'right-half_cons',
                'right-all_cons', 'right-half_acc-half', 'right-all_acc-half', 'right-half_acc-all',
                'Record_trajectory']

    # Load the occupant injury prediction model.
    model_InjPre = RNN(in_dim=16, hid_dim=32, n_layers=2, flag_LSTM=True, bidirectional=True, dropout=0.5)
    model_InjPre.load_state_dict(torch.load('para\\DL_InjuryPrediction.pkl'))
    model_InjPre.eval()

    # Load the vehicle image for visualization.
    img_1 = mpimg.imread('para\image\\red.png')
    img_2 = mpimg.imread('para\image\\blue.png')
    img_list = [img_1, img_2]

    # Load the parameters of vehicles and occupants.
    veh_param, occ_param = load_para(opt.profile_inf, opt.case_num)
    (veh_l, veh_w, veh_cgf, veh_cgs, veh_k, veh_m) = veh_param
    (age, belt, female, airbag, mass_r) = occ_param

    # Translate the activation time of IRM algorithm.
    t_act = int(100 - opt.t_act/10)

    # Define the random seed.
    random_seed = [41, 24, 11, ][opt.case_num - 1] + t_act
    np.random.seed(random_seed)

    # Define the two vehicles in the imminent collision scenario.
    if opt.Level == 'S3':
        Veh_1 = Vehicle_S3(opt.case_num, 0, 1, mass_ratio=mass_r[0], age=age[0], belt=belt[0], female=female[0],
                           airbag=airbag[0], r_seed=random_seed)
        Veh_2 = Vehicle_S3(opt.case_num, 0, 2, mass_ratio=mass_r[1], age=age[1], belt=belt[1], female=female[1],
                           airbag=airbag[1], r_seed=random_seed)
    else:
        Veh_1 = Vehicle_S12(opt.case_num, 0, 1, mass_ratio=mass_r[0], age=age[0], belt=belt[0], female=female[0],
                            airbag=airbag[0], r_seed=random_seed)
        Veh_2 = Vehicle_S12(opt.case_num, 0, 2, mass_ratio=mass_r[1], age=age[1], belt=belt[1], female=female[1],
                            airbag=airbag[1], r_seed=random_seed)

    # Predefine some parameters.
    flag_EB, flag_S3 = True, True
    image_list = []
    INJ = - np.ones(2)
    INJ_ = - np.ones((2, 6))
    V1_x_seq, V1_y_seq, V1_theta_seq, V1_v_long_seq, V1_v_lat_seq, V1_a_seq, V1_omega_r_seq, V1_wheel_anlge_seq = [], [], [], [], [], [], [], []
    V2_x_seq, V2_y_seq, V2_theta_seq, V2_v_long_seq, V2_v_lat_seq, V2_a_seq, V2_omega_r_seq, V2_wheel_anlge_seq = [], [], [], [], [], [], [], []
    t_1 = 0
    t_2 = 0

    if opt.no_visualize:
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.ion()
        plt.axis('equal')

    # Simulate the imminent collision scenario with IRM algorithm for 0-2 seconds.
    # Update the time steps in real-time domain.
    # The minimum time interval is 10 ms in the simulation.
    for i in range(len(Veh_1.x)):
        # print(i)

        # Record the vehicle states at time step i.
        V1_x_seq.append(Veh_1.x[t_1])
        V1_y_seq.append(Veh_1.y[t_1])
        V1_theta_seq.append(Veh_1.theta[t_1])
        V1_v_long_seq.append(Veh_1.v_long[t_1])
        V1_v_lat_seq.append(Veh_1.v_lat[t_1])
        V1_a_seq.append(Veh_1.v_long_dot[t_1])
        V1_omega_r_seq.append(Veh_1.omega_r[t_1])
        V1_wheel_anlge_seq.append(Veh_1.wheel_anlge[t_1])

        V2_x_seq.append(Veh_2.x[t_2])
        V2_y_seq.append(Veh_2.y[t_2])
        V2_theta_seq.append(Veh_2.theta[t_2])
        V2_v_long_seq.append(Veh_2.v_long[t_2])
        V2_v_lat_seq.append(Veh_2.v_lat[t_2])
        V2_a_seq.append(Veh_2.v_long_dot[t_2])
        V2_omega_r_seq.append(Veh_2.omega_r[t_2])
        V2_wheel_anlge_seq.append(Veh_2.wheel_anlge[t_2])

        # Make safety decisions based on the IRM algorithm under the different levels.
        if opt.Level == 'EB' and flag_EB:
            if i >= t_act and (i - t_act) % 10 == 0:
                if opt.Ego_V == 1:
                    # Perceive Vehicle_2's states.
                    v2_state = percept(i, V2_x_seq, V2_y_seq, V2_theta_seq, V2_v_long_seq, V2_v_lat_seq, V2_a_seq,
                                       V2_omega_r_seq, V2_wheel_anlge_seq, V1_x_seq, V1_y_seq, r_seed=random_seed)

                    # Decide whether to activate emergency braking (EB).
                    t_1, flag_EB = deci_EB(i, Veh_1, t_1, v2_state,
                                           (V1_x_seq[-1], V1_y_seq[-1], V1_theta_seq[-1], V1_v_long_seq[-1]))

                elif opt.Ego_V == 2:
                    # Perceive Vehicle_2's states.
                    v1_state = percept(i, V1_x_seq, V1_y_seq, V1_theta_seq, V1_v_long_seq, V1_v_lat_seq, V1_a_seq,
                                       V1_omega_r_seq, V1_wheel_anlge_seq, V2_x_seq, V2_y_seq, r_seed=random_seed)

                    # Decide whether to activate emergency braking (EB).
                    t_2, flag_EB = deci_EB(i, Veh_2, t_2, v1_state,
                                           (V2_x_seq[-1], V2_y_seq[-1], V2_theta_seq[-1], V2_v_long_seq[-1]))

        elif opt.Level == 'S1':
            # The ego vehicle updates decisions with the frequency of 10 Hz.
            if i >= t_act and (i - t_act) % 10 == 0:
                if opt.Ego_V == 1:
                    # Perceive Vehicle_2's states.
                    v2_state = percept(i, V2_x_seq, V2_y_seq, V2_theta_seq, V2_v_long_seq, V2_v_lat_seq, V2_a_seq,
                                       V2_omega_r_seq, V2_wheel_anlge_seq, V1_x_seq, V1_y_seq, r_seed=random_seed)
                    # Make safety decisions.
                    Veh_1.decision(1, i, t_1, v2_state, veh_param, Deci_set, model_InjPre)
                    # Make motion planning.
                    Veh_1.trajectory(i, t_1)
                    t_1 = 0

                elif opt.Ego_V == 2:
                    # Perceive Vehicle_1's states.
                    v1_state = percept(i, V1_x_seq, V1_y_seq, V1_theta_seq, V1_v_long_seq, V1_v_lat_seq, V1_a_seq,
                                       V1_omega_r_seq, V1_wheel_anlge_seq, V2_x_seq, V2_y_seq, r_seed=random_seed)
                    # Make safety decisions.
                    Veh_2.decision(2, i, t_2, v1_state, veh_param, Deci_set, model_InjPre)
                    # Make motion planning.
                    Veh_2.trajectory(i, t_2)
                    t_2 = 0

        elif opt.Level == 'S2':
            # Vehicle_1 updates decisions with the frequency of 10 Hz.
            if i >= t_act and (i - t_act) % 10 == 0:
                # Perceive Vehicle_2's states.
                v2_state = percept(i, V2_x_seq, V2_y_seq, V2_theta_seq, V2_v_long_seq, V2_v_lat_seq, V2_a_seq,
                                   V2_omega_r_seq, V2_wheel_anlge_seq, V1_x_seq, V1_y_seq, r_seed=random_seed)
                # Make safety decisions.
                Veh_1.decision(1, i, t_1, v2_state, veh_param, Deci_set, model_InjPre)
                # Make motion planning.
                Veh_1.trajectory(i, t_1)
                t_1 = 0

            # Vehicle_2 updates decisions with the frequency of 10 Hz.
            if i >= t_act and (i - t_act) % 10 == int(10 // 2):
                # Perceive Vehicle_1's states.
                v1_state = percept(i, V1_x_seq, V1_y_seq, V1_theta_seq, V1_v_long_seq, V1_v_lat_seq, V1_a_seq,
                                   V1_omega_r_seq, V1_wheel_anlge_seq, V2_x_seq, V2_y_seq, r_seed=random_seed)
                # Make safety decisions.
                Veh_2.decision(2, i, t_2, v1_state, veh_param, Deci_set, model_InjPre)
                # Make motion planning.
                Veh_2.trajectory(i, t_2)
                t_2 = 0

        elif opt.Level == 'S3':
            # Vehicles update decisions with the frequency of 10 Hz.
            if i >= t_act and (i - t_act) % 10 == 0 and flag_S3:
                flag_S3 = deci_S3(flag_S3, i, t_1, Veh_1, Veh_2, veh_param, Deci_set, model_InjPre, r_seed=random_seed)
                t_1, t_2 = 0, 0

        # Visualize the simulation environment.
        if opt.no_visualize:
            plot_env(ax, V1_x_seq, V1_y_seq, V1_theta_seq[-1], V2_x_seq, V2_y_seq, V2_theta_seq[-1], veh_l, veh_w,
                     img_list)
            plt.pause(0.0001)
            if opt.save_gif:
                plt.savefig('image/temp_%s.png' % opt.Level)
                image_list.append(imageio.imread('image/temp_%s.png' % opt.Level))

        t_1 += 1
        t_2 += 1

        if i == 0:
            continue

        # Check whether there is a crash at the time step i.
        V1_state = (V1_x_seq[-1], V1_y_seq[-1], V1_theta_seq[-1], V1_x_seq[-2], V1_y_seq[-2], V1_theta_seq[-2])
        V2_state = (V2_x_seq[-1], V2_y_seq[-1], V2_theta_seq[-1], V2_x_seq[-2], V2_y_seq[-2], V2_theta_seq[-2])
        veh_striking_list = det_crash(veh_l, veh_w, V1_state, V2_state)

        # If crash happens, estimate the collision condition and predict occupant injury severity.
        if veh_striking_list:
            delta_v1, delta_v2, delta_v_index = Collision_cond(veh_striking_list, V1_v_long_seq[-1], V2_v_long_seq[-1],
                                                               V2_theta_seq[-1] - V1_theta_seq[-1], veh_param)

            dV_list = [delta_v1, delta_v2]
            angle_list = [np.rad2deg(V2_theta_seq[-1] - V1_theta_seq[-1]), np.rad2deg(V1_theta_seq[-1] - V2_theta_seq[-1])]
            PoI_list = [veh_striking_list[delta_v_index][1], veh_striking_list[delta_v_index][2]]
            Veh_list = [Veh_1, Veh_2]

            for num_i in range(2):
                # Process input valuables of the injury prediction model.
                d_V = round(dV_list[num_i] * 3.6)
                angle = angle_list[num_i] if angle_list[num_i] >= 0 else angle_list[num_i] + 360
                angle = round((angle + 2.5) // 5) * 5
                angle = 0 if angle == 360 else angle
                veh_i = Veh_list[num_i]
                model_input = torch.from_numpy(np.array([[d_V, angle, PoI_list[num_i], PoI_list[1 - num_i], veh_i.age,
                                                          veh_i.female, veh_i.belt, veh_i.airbag,
                                                          veh_i.mass_ratio, ]])).float()

                # Get human injury information using the data-driven injury prediction model.
                pred = model_InjPre(model_input).detach()
                injury = torch.nn.functional.softmax(pred, dim=1).data.numpy()[0]

                # Translate injury probability into OISS.
                injury_score = (0 * injury[0] + 1.37 * injury[1] + 7.54 * injury[2] + 32.22 * injury[3]) / 32.22
                INJ[num_i] = injury_score

                INJ_[num_i, 0] = pred.data.max(1)[1].numpy()
                INJ_[num_i, 1:5] = injury
                INJ_[num_i, 5] = injury_score

            break

    if opt.no_visualize:
        if opt.save_gif:
            for i in range(50):
                image_list.append(imageio.imread('image/temp_%s.png' % opt.Level))
            imageio.mimsave(
                'image/simulation_%s_%s_%s_%s.gif' % (opt.Level, opt.Ego_V, opt.case_num, t_act),
                image_list, duration=0.03)

        plt.pause(5)
        plt.ioff()
        plt.close()


if __name__ == "__main__":
    main()

