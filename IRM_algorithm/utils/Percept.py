''' This module develops the perception module of AVs. '''


import numpy as np


__author__ = "Qingfan Wang"


def percept(i, V_e_x_seq, V_e_y_seq, V_e_theta_seq, V_e_v_long_seq, V_e_v_lat_seq, V_e_a_seq, V_e_omega_r_seq,
            V_e_wheel_anlge_seq, V_x_seq, V_y_seq, r_seed=False):
    ''' Develop the perception module of AVs. '''

    # Define the random seed.
    np.random.seed(r_seed)

    # Define the perception time delay.
    Time_Perc = 10
    Time_Perc_temp = np.min([Time_Perc, i])

    # At the time step (i), obtain the surrounding vehicle's dynamics at the time step (i-Time_Perc_temp).
    v_e_x0, v_e_y0 = V_e_x_seq[i - Time_Perc_temp], V_e_y_seq[i - Time_Perc_temp]

    # Perception error depends on the relative distance.
    Perc_err_x_std = 0.0005 * np.abs(v_e_x0 - V_x_seq[i - Time_Perc_temp]) + 0.005
    Perc_err_x = np.random.normal(0, Perc_err_x_std, 1)[0]
    Perc_err_y_std = 0.0005 * np.abs(v_e_y0 - V_y_seq[i - Time_Perc_temp]) + 0.005
    Perc_err_y = np.random.normal(0, Perc_err_y_std, 1)[0]

    # Estimate the surrounding vehicle's dynamics at the time step (i), using the constant acceleration model.
    x, y = v_e_x0 + Perc_err_x, v_e_y0 + Perc_err_y
    theta, v_long, v_lat, v_long_dot, omega_r, wheel_anlge = V_e_theta_seq[i - Time_Perc_temp], V_e_v_long_seq[
        i - Time_Perc_temp], V_e_v_lat_seq[i - Time_Perc_temp], V_e_a_seq[i - Time_Perc_temp], V_e_omega_r_seq[
        i - Time_Perc_temp], V_e_wheel_anlge_seq[i - Time_Perc_temp]

    # Define the vehicle parameters.
    (l_1, l_2, m, I_z, h, r_wheel) = (1.421, 1.434, 2270, 4600, 0.647, 0.351)
    (C_alhpa, F_x_max, F_y_1_max, F_y_2_max, mu_max, T_max, F_x_Tmax) = (
        100000, 20000, 10400, 10600, 0.88, 3000, 3000 / 0.351)


    x_list, y_list, theta_list, v_long_list, v_lat_list, v_long_dot_list, omega_r_list, wheel_anlge_list = [], [], [], [], [], [], [], []
    x_list.append(x)
    y_list.append(y)
    theta_list.append(theta)
    v_long_list.append(v_long)
    v_lat_list.append(v_lat)
    v_long_dot_list.append(v_long_dot)
    omega_r_list.append(omega_r)
    wheel_anlge_list.append(wheel_anlge)

    # Update vehicle dynamics using the defined plane bicycle model.
    time_step = 0.01
    for time_i in range(int(1.99 / time_step) + Time_Perc_temp):
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

        # Assume the two control inputs (v_long_dot and wheel_anlge) are constant values.
        v_long_dot = v_long_dot
        wheel_anlge = wheel_anlge

        v_long += v_long_dot * time_step

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

    if i < 2:
        v_state = (x_list, y_list, theta_list, v_long_list, x_list[0], y_list[0], theta_list[0], 0)
    else:
        v_state = (x_list, y_list, theta_list, v_long_list, V_e_x_seq[i - Time_Perc_temp - 1] + Perc_err_x,
                   V_e_y_seq[i - Time_Perc_temp - 1] + Perc_err_y, V_e_theta_seq[i - Time_Perc_temp - 1],
                   Time_Perc_temp)

    return v_state
