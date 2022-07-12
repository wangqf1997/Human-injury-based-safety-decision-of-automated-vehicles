# -*- coding: utf-8 -*-
''' Develop the vehicle simulation model, including the decision making module and motion planning module. '''


import torch
import numpy as np
import xlrd

from utils.Veh_dyn import vehicle_dyn
from utils.Det_crash import det_crash
from utils.Con_est import Collision_cond


__author__ = "Qingfan Wang"


class Vehicle_S12:
    ''' Develop the vehicle simulation model under the IRM algorithm of EB, S1, or s2. '''

    def __init__(self, Num_file, Num_ext, Veh_id, mass_ratio, age, belt, airbag, female, r_seed):
        ''' Define the initial parameters and vehicle dynamics. '''

        # Obtain the parameters.
        self.Num_file = Num_file
        self.Num_ext = Num_ext
        self.mass_ratio = mass_ratio
        self.age = age
        self.belt = belt
        self.airbag = airbag
        self.female = female
        self.r_seed = r_seed

        # Initialize the vehicle safety decision.
        self.deci = 'None'

        # Get the reconstructed real-world accident dynamics.
        data_record = xlrd.open_workbook('para/HumanDriver_dynamics_example.xls').sheet_by_name('dynamics')
        if Veh_id == 1:
            self.x = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 0)[1:201])
            self.y = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 1)[1:201])
            self.theta = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 2)[1:201])
            self.v_long = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 3)[1:201])
            self.v_lat = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 4)[1:201])
            self.v_long_dot = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 5)[1:201])
            self.omega_r = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 6)[1:201])
            self.wheel_anlge = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 7)[1:201])
        elif Veh_id == 2:
            self.x = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 9 + 0)[1:201])
            self.y = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 9 + 1)[1:201])
            self.theta = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 9 + 2)[1:201])
            self.v_long = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 9 + 3)[1:201])
            self.v_lat = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 9 + 4)[1:201])
            self.v_long_dot = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 9 + 5)[1:201])
            self.omega_r = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 9 + 6)[1:201])
            self.wheel_anlge = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 9 + 7)[1:201])

    def decision(self, Veh_id, time_i, ego_t, Veh_e_state, veh_param, Deci_set, model):
        ''' Develop the vehicle decision making module. '''

        # Get the vehicle parameters.
        (veh_l, veh_w, veh_cgf, veh_cgs, veh_k, veh_m) = veh_param

        # Obtain the ego vehicle's dynamics.
        veh_state = (self.x[ego_t], self.y[ego_t], self.theta[ego_t], self.v_long[ego_t], self.v_lat[ego_t],
                     self.v_long_dot[ego_t], self.omega_r[ego_t], self.wheel_anlge[ego_t])

        self.injury = []

        ego_t_0 = np.max([ego_t - 1, 0])

        # Use the enumeration method to get the optimal vehicle decision with minimal injuries.
        for deci in Deci_set:
            if deci != 'Record_trajectory':
                x_abs, y_abs, veh_angle, v_long, _, _, _, _ = vehicle_dyn(veh_state, deci, Deci_set)
            elif self.deci in ['None', 'Record_trajectory']:
                x_abs, y_abs, veh_angle, v_long = self.x[ego_t:], self.y[ego_t:], self.theta[ego_t:], self.v_long[
                                                                                                      ego_t:]
            else:
                break

            Veh_x_old, Veh_y_old, Veh_angle_old = self.x[ego_t_0], self.y[ego_t_0], self.theta[ego_t_0]

            # Perceive the surrounding vehicle's dynamics.
            (Veh_e_x, Veh_e_y, Veh_e_angle, Veh_e_V, Veh_e_x_old, Veh_e_y_old, Veh_e_angle_old,
             Time_Perc_temp) = Veh_e_state

            x_abs = np.append(self.x[(ego_t - Time_Perc_temp):ego_t], x_abs)
            y_abs = np.append(self.y[(ego_t - Time_Perc_temp):ego_t], y_abs)
            veh_angle = np.append(self.theta[(ego_t - Time_Perc_temp):ego_t], veh_angle)
            v_long = np.append(self.v_long[(ego_t - Time_Perc_temp):ego_t], v_long)

            # Update the time steps in virtual-time domain.
            for i in range(200 - time_i):
                V1_state_ = (x_abs[i], y_abs[i], veh_angle[i], Veh_x_old, Veh_y_old, Veh_angle_old)
                V2_state_ = (Veh_e_x[i], Veh_e_y[i], Veh_e_angle[i], Veh_e_x_old, Veh_e_y_old, Veh_e_angle_old)
                V1_state = V1_state_ if Veh_id == 1 else V2_state_
                V2_state = V1_state_ if Veh_id == 2 else V2_state_

                # Check whether there is a crash at the time step i.
                veh_striking_list = det_crash(veh_l, veh_w, V1_state, V2_state)

                Veh_x_old, Veh_y_old, Veh_angle_old = x_abs[i], y_abs[i], veh_angle[i]
                Veh_e_x_old, Veh_e_y_old, Veh_e_angle_old = Veh_e_x[i], Veh_e_y[i], Veh_e_angle[i]

                # If crash happens, estimate the collision condition and predict occupant injury severity.
                if veh_striking_list:
                    if Veh_id == 1:
                        V1_v = v_long[i]
                        V2_v = Veh_e_V[i]
                        delta_angle = Veh_e_angle[i] - veh_angle[i]
                    else:
                        V1_v = Veh_e_V[i]
                        V2_v = v_long[i]
                        delta_angle = veh_angle[i] - Veh_e_angle[i]

                    # Obtain delta-v based on the plane 2-DOF rigid-body collision model with momentum conservation.
                    delta_v1, delta_v2, delta_v_index = Collision_cond(veh_striking_list, V1_v, V2_v, delta_angle,
                                                                       veh_param)

                    # Process input valuables of the injury prediction model.
                    d_V = round([delta_v1, delta_v2][Veh_id - 1] * 3.6)
                    angle = np.rad2deg(delta_angle)
                    angle = angle if angle >= 0 else angle + 360
                    angle = round((angle + 2.5) // 5) * 5
                    angle = 0 if angle == 360 else angle
                    PoI_ego, PoI_oth = veh_striking_list[delta_v_index][Veh_id], veh_striking_list[delta_v_index][
                        3 - Veh_id]

                    model_input = torch.from_numpy(np.array([[d_V, angle, PoI_ego, PoI_oth, self.age, self.female,
                                                              self.belt, self.airbag, self.mass_ratio]])).float()

                    # Get human injury information using the data-driven injury prediction model.
                    pred = model(model_input).detach()
                    injury = torch.nn.functional.softmax(pred, dim=1).data.numpy()[0]

                    # Translate injury probability into OISS.
                    injury = (0 * injury[0] + 1.37 * injury[1] + 7.54 * injury[2] + 32.22 * injury[3]) / 32.22
                    self.injury.append(injury)

                    break

                # If no crash happens, OISS will be recorded as zero.
                if i == 200 - time_i - 1:
                    length = np.min([len(x_abs), len(Veh_e_x), len(y_abs), len(Veh_e_y)])
                    distance_min = np.min(
                        ((x_abs[:length] - Veh_e_x[:length]) ** 2 + (y_abs[:length] - Veh_e_y[:length]) ** 2) ** 0.5)
                    self.injury.append(0 - distance_min / 10000)

            if self.injury[-1] == 0:
                if distance_min > 8:
                    break

        # Get the optimal vehicle decision u* with minimal injuries.
        self.deci = Deci_set[self.injury.index(min(self.injury))]

    def trajectory(self, time_i, ego_t):
        ''' Develop the vehicle motion planning module. '''

        veh_state = (self.x[ego_t], self.y[ego_t], self.theta[ego_t], self.v_long[ego_t], self.v_lat[ego_t],
                     self.v_long_dot[ego_t], self.omega_r[ego_t], self.wheel_anlge[ego_t])

        # Make motion planning according to the optimal vehicle decision u*.
        if self.deci != 'Record_trajectory':
            self.x, self.y, self.theta, self.v_long, self.v_lat, self.v_long_dot, self.omega_r, self.wheel_anlge = \
                vehicle_dyn(veh_state, self.deci, if_error=True, r_seed=self.r_seed + time_i)
        else:
            self.x, self.y, self.theta, self.v_long, self.v_lat, self.v_long_dot, self.omega_r, self.wheel_anlge = \
                self.x[ego_t:], self.y[ego_t:], self.theta[ego_t:], self.v_long[ego_t:], self.v_lat[ego_t:], \
                self.v_long_dot[ego_t:], self.omega_r[ego_t:], self.wheel_anlge[ego_t:]


def deci_EB(time_i, Veh_i, t_i, Veh_e_state, Veh_state):
    ''' Develop the vehicle decision making module that decides whether to activate emergency braking (EB). '''
    (Veh_e_x, Veh_e_y, Veh_e_theta, Veh_e_v, _, _, _, Veh_e_t) = Veh_e_state
    (Veh_x, Veh_y, Veh_theta, Veh_v) = Veh_state
    flag_EB = True

    x_rela = (Veh_e_x[Veh_e_t] - Veh_x) * np.cos(Veh_theta) + (Veh_e_y[Veh_e_t] - Veh_y) * np.sin(Veh_theta)
    y_rela = (Veh_e_y[Veh_e_t] - Veh_y) * np.cos(Veh_theta) - (Veh_e_x[Veh_e_t] - Veh_x) * np.sin(Veh_theta)
    t_rela = np.arctan(y_rela / x_rela) if x_rela >= 0 else (np.arctan(y_rela / x_rela) + np.pi)
    V_rela = Veh_v - Veh_e_v[Veh_e_t] * np.cos(Veh_e_theta[Veh_e_t] - Veh_theta)
    TTC = x_rela / V_rela
    if 0 < x_rela < 50 and -45 < np.rad2deg(t_rela) < 45 and 0 < TTC <= 1.4:
        Veh_i.deci = 'straight_dec-all'
        Veh_i.trajectory(time_i, t_i)
        t_i = 0
        flag_EB = False

    return t_i, flag_EB


class Vehicle_S3:
    ''' Develop the vehicle simulation model under the IRM algorithm of S3. '''

    def __init__(self, Num_file, Num_ext, Veh_id, mass_ratio, age, belt, airbag, female, r_seed):
        ''' Define the initial parameters and vehicle dynamics. '''

        # Obtain the relative parameters.
        self.Num_file = Num_file
        self.Num_ext = Num_ext
        self.mass_ratio = mass_ratio
        self.age = age
        self.belt = belt
        self.airbag = airbag
        self.female = female
        self.r_seed = r_seed

        # Initialize the vehicle safety decision.
        self.deci = 'None'

        # Get the reconstructed real-world accident dynamics.
        data_record = xlrd.open_workbook('para/HumanDriver_dynamics_example.xls').sheet_by_name('dynamics')
        if Veh_id == 1:
            self.x = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 0)[1:201])
            self.y = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 1)[1:201])
            self.theta = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 2)[1:201])
            self.v_long = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 3)[1:201])
            self.v_lat = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 4)[1:201])
            self.v_long_dot = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 5)[1:201])
            self.omega_r = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 6)[1:201])
            self.wheel_anlge = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 7)[1:201])
        elif Veh_id == 2:
            self.x = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 9 + 0)[1:201])
            self.y = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 9 + 1)[1:201])
            self.theta = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 9 + 2)[1:201])
            self.v_long = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 9 + 3)[1:201])
            self.v_lat = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 9 + 4)[1:201])
            self.v_long_dot = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 9 + 5)[1:201])
            self.omega_r = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 9 + 6)[1:201])
            self.wheel_anlge = np.array(data_record.row_values(72 * self.Num_file + 18 * self.Num_ext + 9 + 7)[1:201])


def deci_S3(flag_S3, time_i, t_t, Veh_1, Veh_2, veh_param, Deci_set, model, r_seed):
    ''' Develop the vehicle decision making module and motion planning module under the IRM algorithm of S3. '''

    # Get the vehicle parameters.
    (veh_l, veh_w, veh_cgf, veh_cgs, veh_k, veh_m) = veh_param

    # Obtain the vehicles' dynamics.
    veh_1_state = (Veh_1.x[t_t], Veh_1.y[t_t], Veh_1.theta[t_t], Veh_1.v_long[t_t], Veh_1.v_lat[t_t],
                   Veh_1.v_long_dot[t_t], Veh_1.omega_r[t_t], Veh_1.wheel_anlge[t_t])
    veh_2_state = (Veh_2.x[t_t], Veh_2.y[t_t], Veh_2.theta[t_t], Veh_2.v_long[t_t], Veh_2.v_lat[t_t],
                   Veh_2.v_long_dot[t_t], Veh_2.omega_r[t_t], Veh_2.wheel_anlge[t_t])

    Injury1, Injury2 = [], []
    Injury_mean = []
    Deci_1, Deci_2 = [], []

    t_t_0 = np.max([t_t - 1, 0])

    # Use the enumeration method to get the optimal vehicle decision with minimal injuries.
    for deci_1 in Deci_set:
        if deci_1 != 'Record_trajectory':
            x_1_abs, y_1_abs, veh_1_angle, v_1_long, _, _, _, _ = vehicle_dyn(veh_1_state, deci_1)
        elif Veh_1.deci in ['None', 'Record_trajectory']:
            x_1_abs, y_1_abs, veh_1_angle, v_1_long = Veh_1.x[t_t:], Veh_1.y[t_t:], Veh_1.theta[t_t:], Veh_1.v_long[
                                                                                                       t_t:]
        else:
            break

        for deci_2 in Deci_set:
            if deci_2 != 'Record_trajectory':
                x_2_abs, y_2_abs, veh_2_angle, v_2_long, _, _, _, _ = vehicle_dyn(veh_2_state, deci_2)
            elif Veh_2.deci in ['None', 'Record_trajectory']:
                x_2_abs, y_2_abs, veh_2_angle, v_2_long = Veh_2.x[t_t:], Veh_2.y[t_t:], Veh_2.theta[t_t:], Veh_2.v_long[
                                                                                                           t_t:]
            else:
                break

            Veh_1_x_old, Veh_1_y_old, Veh_1_angle_old = Veh_1.x[t_t_0], Veh_1.y[t_t_0], Veh_1.theta[t_t_0]
            Veh_2_x_old, Veh_2_y_old, Veh_2_angle_old = Veh_2.x[t_t_0], Veh_2.y[t_t_0], Veh_2.theta[t_t_0]

            # Update the time steps in virtual-time domain.
            for i in range(200 - time_i):
                V1_state = (x_1_abs[i], y_1_abs[i], veh_1_angle[i], Veh_1_x_old, Veh_1_y_old, Veh_1_angle_old)
                V2_state = (x_2_abs[i], y_2_abs[i], veh_2_angle[i], Veh_2_x_old, Veh_2_y_old, Veh_2_angle_old)

                # Check whether there is a crash at the time step i.
                veh_striking_list = det_crash(veh_l, veh_w, V1_state, V2_state)

                Veh_1_x_old, Veh_1_y_old, Veh_1_angle_old = x_1_abs[i], y_1_abs[i], veh_1_angle[i]
                Veh_2_x_old, Veh_2_y_old, Veh_2_angle_old = x_2_abs[i], y_2_abs[i], veh_2_angle[i]

                # If crash happens, estimate the collision condition and predict occupant injury severity.
                if veh_striking_list:
                    delta_v1, delta_v2, delta_v_index = Collision_cond(veh_striking_list, v_1_long[i], v_2_long[i],
                                                                       veh_2_angle[i] - veh_1_angle[i], veh_param)

                    d_V_list = [delta_v1, delta_v2]
                    angle_list = [np.rad2deg(veh_2_angle[i] - veh_1_angle[i]),
                                  np.rad2deg(veh_1_angle[i] - veh_2_angle[i])]
                    PoI_list = [veh_striking_list[delta_v_index][1], veh_striking_list[delta_v_index][2]]
                    injury_list = [0, 0]
                    Veh_list = [Veh_1, Veh_2]

                    for num_i in range(2):
                        # Process input valuables of the injury prediction model.
                        d_V = round(d_V_list[num_i] * 3.6)
                        angle_i = angle_list[num_i] if angle_list[num_i] >= 0 else angle_list[num_i] + 360
                        angle_i = round((angle_i + 2.5) // 5) * 5
                        angle_i = 0 if angle_i == 360 else angle_i
                        veh_i = Veh_list[num_i]

                        model_input = torch.from_numpy(np.array([[d_V, angle_i, PoI_list[num_i], PoI_list[1 - num_i],
                                                                  veh_i.age, veh_i.female, veh_i.belt, veh_i.airbag,
                                                                  veh_i.mass_ratio, ]])).float()

                        # Get human injury information using the data-driven injury prediction model.
                        pred = model(model_input).detach()
                        injury = torch.nn.functional.softmax(pred, dim=1).data.numpy()[0]

                        # Translate injury probability into OISS.
                        injury = (0 * injury[0] + 1.37 * injury[1] + 7.54 * injury[2] + 32.22 * injury[3]) / 32.22
                        injury_list[num_i] = injury

                    Injury1.append(injury_list[0])
                    Injury2.append(injury_list[1])
                    Injury_mean.append((injury_list[0] + injury_list[1])/2)
                    Deci_1.append(deci_1)
                    Deci_2.append(deci_2)

                    break

                # If no crash happens, OISS will be recorded as zero.
                if i == 200 - time_i - 1:
                    length = np.min([len(x_1_abs), len(y_1_abs), len(x_2_abs), len(y_2_abs)])
                    distance_min = np.min(((np.array(x_1_abs[:length]) - np.array(x_2_abs[:length])) ** 2 +
                                           (np.array(y_1_abs[:length]) - np.array(y_2_abs[:length])) ** 2) ** 0.5)
                    Injury1.append(0 - distance_min / 10000)
                    Injury2.append(0 - distance_min / 10000)
                    Injury_mean.append(0 - distance_min / 10000)
                    Deci_1.append(deci_1)
                    Deci_2.append(deci_2)

            if Injury_mean[-1] <= 0:
                if distance_min > 5:
                    break
        if Injury_mean[-1] <= 0:
            if distance_min > 5:
                break

    # Get the optimal vehicle decision u* with minimal injuries.
    Veh_1.deci = Deci_1[Injury_mean.index(min(Injury_mean))]
    Veh_2.deci = Deci_2[Injury_mean.index(min(Injury_mean))]
    Veh_1.injury = Injury1[Injury_mean.index(min(Injury_mean))]
    Veh_2.injury = Injury2[Injury_mean.index(min(Injury_mean))]

    if Veh_1.deci != 'Record_trajectory':
        Veh_1.x, Veh_1.y, Veh_1.theta, Veh_1.v_long, Veh_1.v_lat, Veh_1.v_long_dot, Veh_1.omega_r, Veh_1.wheel_anlge = \
            vehicle_dyn(veh_1_state, Veh_1.deci, if_error=True, r_seed=r_seed + time_i)
    else:
        Veh_1.x, Veh_1.y, Veh_1.theta, Veh_1.v_long, Veh_1.v_lat, Veh_1.v_long_dot, Veh_1.omega_r, Veh_1.wheel_anlge = \
            Veh_1.x[t_t:], Veh_1.y[t_t:], Veh_1.theta[t_t:], Veh_1.v_long[t_t:], Veh_1.v_lat[t_t:], \
            Veh_1.v_long_dot[t_t:], Veh_1.omega_r[t_t:], Veh_1.wheel_anlge[t_t:]
    if Veh_2.deci != 'Record_trajectory':
        Veh_2.x, Veh_2.y, Veh_2.theta, Veh_2.v_long, Veh_2.v_lat, Veh_2.v_long_dot, Veh_2.omega_r, Veh_2.wheel_anlge = \
            vehicle_dyn(veh_2_state, Veh_2.deci, if_error=True, r_seed=r_seed + time_i)
    else:
        Veh_2.x, Veh_2.y, Veh_2.theta, Veh_2.v_long, Veh_2.v_lat, Veh_2.v_long_dot, Veh_2.omega_r, Veh_2.wheel_anlge = \
            Veh_2.x[t_t:], Veh_2.y[t_t:], Veh_2.theta[t_t:], Veh_2.v_long[t_t:], Veh_2.v_lat[t_t:], \
            Veh_2.v_long_dot[t_t:], Veh_2.omega_r[t_t:], Veh_2.wheel_anlge[t_t:]

    if min(Injury_mean) <= 0:
        flag_S3 = False

    return flag_S3
