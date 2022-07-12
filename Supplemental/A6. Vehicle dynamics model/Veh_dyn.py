'''
-------------------------------------------------------------------------------------------------
This code accompanies the paper titled "Human injury-based safety decision of automated vehicles"
Author: Qingfan Wang, Qing Zhou, Miao Lin, Bingbing Nie
Corresponding author: Bingbing Nie (nbb@tsinghua.edu.cn)
-------------------------------------------------------------------------------------------------
'''


import numpy as np


__author__ = "Qingfan Wang"


def vehicle_dyn(veh_state, veh_deci, if_error=False, r_seed=False):
    ''' Model the vehicle dynamics and constraints. '''

    # Define the vehicle parameters.
    (x, y, theta, v_long, v_lat, v_long_dot, omega_r, wheel_anlge) = veh_state
    (l_1, l_2, m, I_z, h, r_wheel) = (1.421, 1.434, 2270, 4600, 0.647, 0.351)
    (C_alhpa, F_x_max, F_y_1_max, F_y_2_max, mu_max, T_max, F_x_Tmax) = (
        100000, 20000, 10400, 10600, 0.88, 3000, 3000 / 0.351)

    # Get maximum acceleration or deceleration under the decision u.
    if 'dec-all' in veh_deci:
        v_long_dot_max = -8
    elif 'dec-half' in veh_deci:
        v_long_dot_max = -4
    elif 'cons' in veh_deci:
        v_long_dot_max = 0
    elif 'acc-half' in veh_deci:
        v_long_dot_max = 4
    elif 'acc-all' in veh_deci:
        v_long_dot_max = 8

    # Get maximum steering angle under the decision u.
    wheel_anlge_max_ = np.max([(25 - v_long) / 25 * 15 + 5, 5])
    if 'left-all' in veh_deci:
        wheel_anlge_max = np.deg2rad(wheel_anlge_max_)
    elif 'left-half' in veh_deci:
        wheel_anlge_max = np.deg2rad(wheel_anlge_max_ / 2)
    elif 'straight' in veh_deci:
        wheel_anlge_max = 0
    elif 'right-half' in veh_deci:
        wheel_anlge_max = np.deg2rad(- wheel_anlge_max_ / 2)
    elif 'right-all' in veh_deci:
        wheel_anlge_max = np.deg2rad(- wheel_anlge_max_)

    x_list, y_list, theta_list, v_long_list, v_lat_list, v_long_dot_list, omega_r_list, wheel_anlge_list = [], [], [], [], [], [], [], []
    x_list.append(x)
    y_list.append(y)
    theta_list.append(theta)
    v_long_list.append(v_long)
    v_lat_list.append(v_lat)
    v_long_dot_list.append(v_long_dot)
    omega_r_list.append(omega_r)
    wheel_anlge_list.append(wheel_anlge)

    # Define the random seed.
    np.random.seed(r_seed)

    # Update vehicle dynamics using the defined plane bicycle model.
    time_step = 0.01
    for time_i in range(int(1.99/time_step)):
        if v_long_dot < 0.1 and v_long < 0.02:
            x_list.append(x)
            y_list.append(y)
            theta_list.append(theta)
            v_long_list.append(v_long)
            v_lat_list.append(v_lat)
            v_long_dot_list.append(v_long_dot)
            omega_r_list.append(omega_r)
            wheel_anlge_list.append(wheel_anlge)
            continue

        beta = np.arctan(v_lat / v_long)

        alpha_1 = - (beta + l_1 * omega_r / v_long - wheel_anlge)
        alpha_2 = - (beta - l_1 * omega_r / v_long)

        # Define the simplified linear vehicle tire model with saturation.
        F_y_1 = np.min([C_alhpa * np.abs(alpha_1), C_alhpa * np.deg2rad(8)]) * np.sign(alpha_1)
        F_y_2 = np.min([C_alhpa * np.abs(alpha_2), C_alhpa * np.deg2rad(8)]) * np.sign(alpha_2)

        omega_r_dot = (l_1 * F_y_1 * np.cos(wheel_anlge) - l_2 * F_y_2) / I_z
        v_lat_dot = (F_y_1 * np.cos(wheel_anlge) + F_y_2) / m - v_long * wheel_anlge
        F_x = m * (v_long_dot - v_lat * wheel_anlge)

        omega_r += omega_r_dot * time_step
        v_lat += v_lat_dot * time_step

        # Define control errors in the vehicle longitudinal dynamics.
        Cont_error_long = np.random.normal(0, 2, 1)[0] if if_error else 0
        v_long_dot_dot = 20 + Cont_error_long
        if 0 <= v_long_dot_max - v_long_dot <= v_long_dot_dot * time_step or 0 >= v_long_dot_max - v_long_dot >= v_long_dot_dot * time_step:
            v_long_dot = v_long_dot_max
        else:
            v_long_dot += v_long_dot_dot * time_step if v_long_dot < v_long_dot_max else -(
                    v_long_dot_dot + 5) * time_step
        v_long += v_long_dot * time_step

        # Define control errors in the vehicle lateral dynamics.
        Cont_error_lat = np.random.normal(0, 2, 1)[0] if if_error else 0
        wheel_anlge_dot = np.deg2rad(15 + Cont_error_lat)
        if 0 <= wheel_anlge_max - wheel_anlge <= wheel_anlge_dot * time_step or 0 >= wheel_anlge_max - wheel_anlge >= wheel_anlge_dot * time_step:
            wheel_anlge = wheel_anlge_max
        else:
            wheel_anlge += wheel_anlge_dot * time_step if wheel_anlge < wheel_anlge_max else -wheel_anlge_dot * time_step

        theta += omega_r * time_step
        x += (v_long * np.cos(theta) - v_lat * np.sin(theta)) * time_step
        y += (v_lat * np.cos(theta) + v_long * np.sin(theta)) * time_step

        x_list.append(x)
        y_list.append(y)
        theta_list.append(theta)
        v_long_list.append(v_long)
        v_lat_list.append(v_lat)
        v_long_dot_list.append(v_long_dot)
        omega_r_list.append(omega_r)
        wheel_anlge_list.append(wheel_anlge)

        # Define vehicle tire force constraints based on the laws of physics (friction ellipse).
        mu_1 = (F_x / F_x_max) ** 2 + (F_y_1 / F_y_1_max / (1 - F_x * h / (m * 9.8) / l_1)) ** 2
        mu_2 = (F_x / F_x_max) ** 2 + (F_y_2 / F_y_2_max / (1 + F_x * h / (m * 9.8) / l_2)) ** 2
        if mu_1 > mu_max or mu_2 > mu_max or F_x > F_x_Tmax:
            if v_long_dot > 0:
                v_long_dot -= 0.5
            else:
                v_long_dot += 0.5

    return x_list, y_list, theta_list, v_long_list, v_lat_list, v_long_dot_list, omega_r_list, wheel_anlge_list
