''' This module develops the collision condition estimation model. '''


import torch
import numpy as np
from torch import nn
from torch.nn.utils import weight_norm


__author__ = "Qingfan Wang"


def Collision_cond(veh_striking_list, V1_v, V2_v, delta_angle, veh_param):
    ''' Estimate the collision condition. '''

    (veh_l, veh_w, veh_cgf, veh_cgs, veh_k, veh_m) = veh_param

    delta_angle_2 = np.arccos(np.abs(np.cos(delta_angle)))
    if -1e-6 < delta_angle_2 < 1e-6:
        delta_angle_2 = 1e-6

    delta_v1_list = []
    delta_v2_list = []

    # Estimate the collision condition (delat-v) according to the principal impact direction.
    for veh_striking in veh_striking_list:
        if veh_striking[0] == 1:
            veh_ca = np.arctan(veh_cgf[0] / veh_cgs[0])
            veh_a2 = np.abs(veh_cgs[1] - veh_striking[3])
            veh_RDS = np.abs(V1_v * np.cos(delta_angle) - V2_v)
            veh_a1 = np.abs(np.sqrt(veh_cgf[0] ** 2 + veh_cgs[0] ** 2) * np.cos(veh_ca + delta_angle_2))
            if (veh_striking[1]+1) in [16, 1, 2, 3, 17, 20, 21] and (veh_striking[2]+1) in [16, 1, 2, 3, 17, 20, 21]:
                veh_e = 2 / veh_RDS
            else:
                veh_e = 0.5 / veh_RDS

        elif veh_striking[0] == 2:
            veh_ca = np.arctan(veh_cgf[0] / veh_cgs[0])
            veh_a2 = np.abs(veh_cgf[1] - veh_striking[3])
            veh_a1 = np.abs(np.sqrt(veh_cgf[0] ** 2 + veh_cgs[0] ** 2) * np.cos(delta_angle_2 - veh_ca + np.pi / 2))
            veh_RDS = V1_v * np.sin(delta_angle_2)
            veh_e = 1.5 / veh_RDS

        elif veh_striking[0] == 3:
            veh_ca = np.arctan(veh_cgf[1] / veh_cgs[1])
            veh_a1 = np.abs(veh_cgs[0] - veh_striking[3])
            veh_RDS = np.abs(V2_v * np.cos(delta_angle) - V1_v)
            veh_a2 = np.abs(np.sqrt(veh_cgf[1] ** 2 + veh_cgs[1] ** 2) * np.cos(veh_ca + delta_angle_2))
            if (veh_striking[1]+1) in [16, 1, 2, 3, 17, 20, 21] and (veh_striking[2]+1) in [16, 1, 2, 3, 17, 20, 21]:
                veh_e = 2 / veh_RDS
            else:
                veh_e = 0.5 / veh_RDS

        elif veh_striking[0] == 4:
            veh_ca = np.arctan(veh_cgf[1] / veh_cgs[1])
            veh_a1 = np.abs(veh_cgf[0] - veh_striking[3])
            veh_a2 = np.abs(np.sqrt(veh_cgf[1] ** 2 + veh_cgs[1] ** 2) * np.cos(delta_angle_2 - veh_ca + np.pi / 2))
            veh_RDS = V2_v * np.sin(delta_angle_2)
            veh_e = 1.5 / veh_RDS


        # Obtain delta-v based on the plane 2-DOF rigid-body collision model with momentum conservation.
        veh_y1 = veh_k[0] ** 2 / (veh_a1 ** 2 + veh_k[0] ** 2)
        veh_y2 = veh_k[1] ** 2 / (veh_a2 ** 2 + veh_k[1] ** 2)
        delta_v1 = (1 + veh_e) * veh_m[1] * veh_y1 * veh_y2 * veh_RDS / (veh_m[0] * veh_y1 + veh_m[1] * veh_y2)
        delta_v2 = (1 + veh_e) * veh_m[0] * veh_y1 * veh_y2 * veh_RDS / (veh_m[0] * veh_y1 + veh_m[1] * veh_y2)

        delta_v1_list.append(delta_v1)
        delta_v2_list.append(delta_v2)

    delta_v1_ = max(delta_v1_list)
    delta_v2_ = max(delta_v2_list)
    index = delta_v1_list.index(max(delta_v1_list))

    return delta_v1_, delta_v2_, index