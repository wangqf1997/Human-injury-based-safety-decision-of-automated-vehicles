# -*- coding: utf-8 -*-
''' This module develops the collision detection module. '''


import matplotlib.pyplot as plt
import numpy as np


__author__ = "Qingfan Wang"


def det_crash(veh_l, veh_w, V1_state, V2_state):
    ''' Develop the collision detection module. '''

    crash_condition = []
    A, B, C, D, E, F = 17, 18, 19, 20, 21, 22

    # Get the vehicle dynamics.
    (v1_x, v1_y, v1_angle, v1_x_old, v1_y_old, v1_angle_old) = V1_state
    (v2_x, v2_y, v2_angle, v2_x_old, v2_y_old, v2_angle_old) = V2_state

    # Get four vertexes' coordinates of the vehicle I.
    v1_x1 = v1_x + (veh_l[0] / 2) * np.cos(v1_angle) - (veh_w[0] / 2) * np.sin(v1_angle)
    v1_x2 = v1_x + (veh_l[0] / 2) * np.cos(v1_angle) + (veh_w[0] / 2) * np.sin(v1_angle)
    v1_x3 = v1_x - (veh_l[0] / 2) * np.cos(v1_angle) + (veh_w[0] / 2) * np.sin(v1_angle)
    v1_x4 = v1_x - (veh_l[0] / 2) * np.cos(v1_angle) - (veh_w[0] / 2) * np.sin(v1_angle)
    v1_y1 = v1_y + (veh_l[0] / 2) * np.sin(v1_angle) + (veh_w[0] / 2) * np.cos(v1_angle)
    v1_y2 = v1_y + (veh_l[0] / 2) * np.sin(v1_angle) - (veh_w[0] / 2) * np.cos(v1_angle)
    v1_y3 = v1_y - (veh_l[0] / 2) * np.sin(v1_angle) - (veh_w[0] / 2) * np.cos(v1_angle)
    v1_y4 = v1_y - (veh_l[0] / 2) * np.sin(v1_angle) + (veh_w[0] / 2) * np.cos(v1_angle)

    # Get four vertexes' coordinates of the vehicle II.
    v2_x1 = v2_x + (veh_l[1] / 2) * np.cos(v2_angle) - (veh_w[1] / 2) * np.sin(v2_angle)
    v2_x2 = v2_x + (veh_l[1] / 2) * np.cos(v2_angle) + (veh_w[1] / 2) * np.sin(v2_angle)
    v2_x3 = v2_x - (veh_l[1] / 2) * np.cos(v2_angle) + (veh_w[1] / 2) * np.sin(v2_angle)
    v2_x4 = v2_x - (veh_l[1] / 2) * np.cos(v2_angle) - (veh_w[1] / 2) * np.sin(v2_angle)
    v2_y1 = v2_y + (veh_l[1] / 2) * np.sin(v2_angle) + (veh_w[1] / 2) * np.cos(v2_angle)
    v2_y2 = v2_y + (veh_l[1] / 2) * np.sin(v2_angle) - (veh_w[1] / 2) * np.cos(v2_angle)
    v2_y3 = v2_y - (veh_l[1] / 2) * np.sin(v2_angle) - (veh_w[1] / 2) * np.cos(v2_angle)
    v2_y4 = v2_y - (veh_l[1] / 2) * np.sin(v2_angle) + (veh_w[1] / 2) * np.cos(v2_angle)

    # Calculate the relative angle.
    delta_angle_0 = np.rad2deg(v1_angle - v2_angle)
    delta_angle_0 = delta_angle_0 if delta_angle_0 >= 0 else delta_angle_0 + 360
    delta_angle = ((delta_angle_0 + 2.5) // 5) * 5
    delta_angle = 0 if delta_angle == 360 else delta_angle

    # Detect four vertexes of Vehicle I in turn.
    num = -1
    for (x, y) in [(v1_x1, v1_y1), (v1_x2, v1_y2), (v1_x3, v1_y3), (v1_x4, v1_y4)]:
        num += 1

        # Calculate the relative positions.
        x_ = x - v2_x
        y_ = y - v2_y
        x__ = x_ * np.cos(v2_angle) + y_ * np.sin(v2_angle)
        y__ = y_ * np.cos(v2_angle) - x_ * np.sin(v2_angle)

        # Detect whether a vertex of Vehicle I touches Vehicle II.
        if np.abs(x__) <= (veh_l[1] / 2) and np.abs(y__) <= (veh_w[1] / 2):
            sign1 = 1 if num < 1.5 else -1
            sign2 = 1 if 0.5 < num < 2.5 else -1

            v1_x_old_ = v1_x_old + sign1 * (veh_l[0] / 2) * np.cos(v1_angle_old) + sign2 * (veh_w[0] / 2) * np.sin(
                v1_angle_old)
            v1_y_old_ = v1_y_old + sign1 * (veh_l[0] / 2) * np.sin(v1_angle_old) - sign2 * (veh_w[0] / 2) * np.cos(
                v1_angle_old)

            # Calculate the relative positions at the previous time step.
            x_old = v1_x_old_ - v2_x_old
            y_old = v1_y_old_ - v2_y_old
            x__old = x_old * np.cos(v2_angle_old) + y_old * np.sin(v2_angle_old)
            y__old = y_old * np.cos(v2_angle_old) - x_old * np.sin(v2_angle_old)

            flag_error = True

            # Determine two vehicles' point of impact (POI).
            if x__old > (veh_l[1] / 2) and x__ < (veh_l[1] / 2):

                if delta_angle in [90, 270]:
                    if (num == 1 and delta_angle == 270) or (num == 3 and delta_angle == 90):
                        y__2 = (y__ + veh_w[1] / 2) / 2
                        if y__2 > 0.25 * veh_w[1]:
                            v2_POI = 16
                        elif y__2 > 0.05 * veh_w[1]:
                            v2_POI = 1
                        elif y__2 > -0.05 * veh_w[1]:
                            v2_POI = E
                        elif y__2 > -0.25 * veh_w[1]:
                            v2_POI = 2
                        else:
                            v2_POI = 3

                        if (num == 1 and delta_angle == 270):
                            x__1 = veh_l[0] / 2 - (veh_w[1] / 2 - y__2)
                            if x__1 > 0.147 * veh_l[0]:
                                v1_POI = 4
                            elif x__1 > -0.0686 * veh_l[0]:
                                v1_POI = 5
                            elif x__1 > -0.284 * veh_l[0]:
                                v1_POI = 6
                            else:
                                v1_POI = 7

                        elif (num == 3 and delta_angle == 90):
                            x__1 = - (veh_l[0] / 2 - (veh_w[1] / 2 - y__2))
                            if x__1 > 0.147 * veh_l[0]:
                                v1_POI = 15
                            elif x__1 > -0.0686 * veh_l[0]:
                                v1_POI = 14
                            elif x__1 > -0.284 * veh_l[0]:
                                v1_POI = 13
                            else:
                                v1_POI = 12

                        distance = (veh_l[0] / 2) - x__1
                        crash_condition.append([4, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False

                    if (num == 0 and delta_angle == 90) or (num == 2 and delta_angle == 270):
                        y__2 = (y__ - veh_w[1] / 2) / 2
                        if y__2 > 0.25 * veh_w[1]:
                            v2_POI = 16
                        elif y__2 > 0.05 * veh_w[1]:
                            v2_POI = 1
                        elif y__2 > -0.05 * veh_w[1]:
                            v2_POI = E
                        elif y__2 > -0.25 * veh_w[1]:
                            v2_POI = 2
                        else:
                            v2_POI = 3

                        if (num == 0 and delta_angle == 90):
                            x__1 = veh_l[0] / 2 - (veh_w[1] / 2 + y__2)
                            if x__1 > 0.147 * veh_l[0]:
                                v1_POI = 15
                            elif x__1 > -0.0686 * veh_l[0]:
                                v1_POI = 14
                            elif x__1 > -0.284 * veh_l[0]:
                                v1_POI = 13
                            else:
                                v1_POI = 12

                        elif (num == 2 and delta_angle == 270):
                            x__1 = - (veh_l[0] / 2 - (veh_w[1] / 2 + y__2))
                            if x__1 > 0.147 * veh_l[0]:
                                v1_POI = 4
                            elif x__1 > -0.0686 * veh_l[0]:
                                v1_POI = 5
                            elif x__1 > -0.284 * veh_l[0]:
                                v1_POI = 6
                            else:
                                v1_POI = 7

                        distance = (veh_l[0] / 2) - x__1
                        crash_condition.append([4, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False

                elif delta_angle == 0:
                    if num in [2, 3]:
                        if num == 2:
                            if y__ + veh_w[0] > veh_w[1] / 2:
                                y__2 = (y__ + veh_w[1] / 2) / 2
                                if len(crash_condition) > 0:
                                    continue
                            else:
                                y__2 = y__ + veh_w[0] / 2
                            y__1 = y__2 - y__ - veh_w[0] / 2

                        elif num == 3:
                            if y__ - veh_w[0] < - veh_w[1] / 2:
                                y__2 = (y__ - veh_w[1] / 2) / 2
                                if len(crash_condition) > 0:
                                    continue
                            else:
                                y__2 = y__ - veh_w[0] / 2
                            y__1 = y__2 - y__ + veh_w[0] / 2

                        if y__1 > 0.25 * veh_w[0]:
                            v1_POI = 11
                        elif y__1 > 0.05 * veh_w[0]:
                            v1_POI = 10
                        elif y__1 > -0.05 * veh_w[0]:
                            v1_POI = F
                        elif y__1 > -0.25 * veh_w[0]:
                            v1_POI = 9
                        else:
                            v1_POI = 8

                        if y__2 > 0.25 * veh_w[1]:
                            v2_POI = 16
                        elif y__2 > 0.05 * veh_w[1]:
                            v2_POI = 1
                        elif y__2 > -0.05 * veh_w[1]:
                            v2_POI = E
                        elif y__2 > -0.25 * veh_w[1]:
                            v2_POI = 2
                        else:
                            v2_POI = 3

                        distance = (veh_w[0] / 2) - np.abs(y__1)
                        crash_condition.append([3, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False

                elif delta_angle == 180:
                    if num in [0, 1]:
                        if num == 0:
                            if y__ + veh_w[0] > veh_w[1] / 2:
                                y__2 = (y__ + veh_w[1] / 2) / 2
                            else:
                                y__2 = y__ + veh_w[0] / 2
                            y__1 = -(y__2 - y__ - veh_w[0] / 2)

                        elif num == 1:
                            if y__ - veh_w[0] < - veh_w[1] / 2:
                                y__2 = (y__ - veh_w[1] / 2) / 2
                            else:
                                y__2 = y__ - veh_w[0] / 2
                            y__1 = -(y__2 - y__ + veh_w[0] / 2)

                        if y__1 > 0.25 * veh_w[0]:
                            v1_POI = 16
                        elif y__1 > 0.05 * veh_w[0]:
                            v1_POI = 1
                        elif y__1 > -0.05 * veh_w[0]:
                            v1_POI = E
                        elif y__1 > -0.25 * veh_w[0]:
                            v1_POI = 2
                        else:
                            v1_POI = 3

                        if y__2 > 0.25 * veh_w[1]:
                            v2_POI = 16
                        elif y__2 > 0.05 * veh_w[1]:
                            v2_POI = 1
                        elif y__2 > -0.05 * veh_w[1]:
                            v2_POI = E
                        elif y__2 > -0.25 * veh_w[1]:
                            v2_POI = 2
                        else:
                            v2_POI = 3

                        distance = (veh_w[1] / 2) - np.abs(y__2)
                        crash_condition.append([1, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                else:
                    v1_POI = [D, A, B, C][num]
                    if (v1_POI == D and 90 < delta_angle < 180) or (v1_POI == A and 180 < delta_angle < 270) or (
                            v1_POI == B and 270 < delta_angle < 360) or (v1_POI == C and 0 < delta_angle < 90):
                        if y__ > 0.25 * veh_w[1]:
                            v2_POI = 16
                        elif y__ > 0:
                            v2_POI = 1
                        elif y__ > -0.25 * veh_w[1]:
                            v2_POI = 2
                        else:
                            v2_POI = 3

                        distance = (veh_w[1] / 2) - np.abs(y__)
                        crash_condition.append([1, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False

            elif x__old < -(veh_l[1] / 2) and x__ > -(veh_l[1] / 2):
                if delta_angle in [90, 180, 270]:
                    pass
                elif delta_angle == 0:
                    if num == 0:
                        if y__ - veh_w[0] < - veh_w[1] / 2:
                            y__2 = (y__ - veh_w[1] / 2) / 2
                        else:
                            y__2 = y__ - veh_w[0] / 2
                        y__1 = y__2 - y__ + veh_w[0] / 2

                    elif num == 1:
                        if y__ + veh_w[0] > veh_w[1] / 2:
                            y__2 = (y__ + veh_w[1] / 2) / 2
                        else:
                            y__2 = y__ + veh_w[0] / 2
                        y__1 = y__2 - y__ - veh_w[0] / 2

                    if y__1 > 0.25 * veh_w[0]:
                        v1_POI = 16
                    elif y__1 > 0.05 * veh_w[0]:
                        v1_POI = 1
                    elif y__1 > -0.05 * veh_w[0]:
                        v1_POI = E
                    elif y__1 > -0.25 * veh_w[0]:
                        v1_POI = 2
                    else:
                        v1_POI = 3

                    if y__2 > 0.25 * veh_w[1]:
                        v2_POI = 11
                    elif y__2 > 0.05 * veh_w[1]:
                        v2_POI = 10
                    elif y__2 > -0.05 * veh_w[1]:
                        v2_POI = F
                    elif y__2 > -0.25 * veh_w[1]:
                        v2_POI = 9
                    else:
                        v2_POI = 8

                    distance = (veh_w[1] / 2) - np.abs(y__2)
                    crash_condition.append([1, v1_POI - 1, v2_POI - 1, distance])
                    flag_error = False
                else:
                    v1_POI = [D, A, B, C][num]
                    if (v1_POI == D and 270 < delta_angle < 360) or (v1_POI == A and 0 < delta_angle < 90):
                        if y__ > 0.25 * veh_w[1]:
                            v2_POI = 11
                        elif y__ > 0:
                            v2_POI = 10
                        elif y__ > -0.25 * veh_w[1]:
                            v2_POI = 9
                        else:
                            v2_POI = 8

                        distance = (veh_w[1] / 2) - np.abs(y__)
                        crash_condition.append([1, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False

            if y__old > (veh_w[1] / 2) and y__ < (veh_w[1] / 2):
                if delta_angle == 90:
                    if num in [2, 3]:
                        if num == 2:
                            if x__ - veh_w[0] < - veh_l[1] / 2:
                                x__2 = (x__ - veh_l[1] / 2) / 2
                            else:
                                x__2 = x__ - veh_w[0] / 2
                            y__1 = - (x__2 - x__ + veh_w[0] / 2)

                        elif num == 3:
                            if x__ + veh_w[0] > veh_l[1] / 2:
                                x__2 = (x__ + veh_l[1] / 2) / 2
                            else:
                                x__2 = x__ + veh_w[0] / 2
                            y__1 = x__2 - x__ - veh_w[0] / 2

                        if y__1 > 0.25 * veh_w[0]:
                            v1_POI = 11
                        elif y__1 > 0.05 * veh_w[0]:
                            v1_POI = 10
                        elif y__1 > -0.05 * veh_w[0]:
                            v1_POI = F
                        elif y__1 > -0.25 * veh_w[0]:
                            v1_POI = 9
                        else:
                            v1_POI = 8

                        if x__2 > 0.147 * veh_l[1]:
                            v2_POI = 15
                        elif x__2 > -0.0686 * veh_l[1]:
                            v2_POI = 14
                        elif x__2 > -0.284 * veh_l[1]:
                            v2_POI = 13
                        else:
                            v2_POI = 12

                        distance = (veh_l[1] / 2) - x__2
                        crash_condition.append([2, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False

                elif delta_angle == 0 or delta_angle == 180:
                    if (num == 1 and 357 < delta_angle_0 < 360) or (num == 2 and 0 < delta_angle_0 < 3) or (
                            num == 3 and 177 < delta_angle_0 < 180) or (num == 0 and 180 < delta_angle_0 < 183):
                        v1_POI = [D, A, B, C][num]
                        if x__ > 0.147 * veh_l[1]:
                            v2_POI = 15
                        elif x__ > -0.0686 * veh_l[1]:
                            v2_POI = 14
                        elif x__ > -0.284 * veh_l[1]:
                            v2_POI = 13
                        else:
                            v2_POI = 12

                        distance = (veh_l[1] / 2) - x__
                        crash_condition.append([2, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False

                elif delta_angle == 270:
                    if num in [0, 1]:
                        if num == 0:
                            if x__ - veh_w[0] < - veh_l[1] / 2:
                                x__2 = (x__ - veh_l[1] / 2) / 2
                            else:
                                x__2 = x__ - veh_w[0] / 2
                            y__1 = x__2 - x__ + veh_w[0] / 2

                        elif num == 1:
                            if x__ + veh_w[0] > veh_l[1] / 2:
                                x__2 = (x__ + veh_l[1] / 2) / 2
                            else:
                                x__2 = x__ + veh_w[0] / 2
                            y__1 = x__2 - x__ - veh_w[0] / 2

                        if y__1 > 0.25 * veh_w[0]:
                            v1_POI = 16
                        elif y__1 > 0.05 * veh_w[0]:
                            v1_POI = 1
                        elif y__1 > -0.05 * veh_w[0]:
                            v1_POI = E
                        elif y__1 > -0.25 * veh_w[0]:
                            v1_POI = 2
                        else:
                            v1_POI = 3

                        if x__2 > 0.147 * veh_l[1]:
                            v2_POI = 15
                        elif x__2 > -0.0686 * veh_l[1]:
                            v2_POI = 14
                        elif x__2 > -0.284 * veh_l[1]:
                            v2_POI = 13
                        else:
                            v2_POI = 12

                        distance = (veh_l[1] / 2) - x__2
                        crash_condition.append([2, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False
                else:
                    if (num == 1 and 180 < delta_angle < 270) or (num == 0 and 270 < delta_angle < 360):
                        flag_error = False
                    else:
                        v1_POI = [D, A, B, C][num]
                        if x__ > 0.147 * veh_l[1]:
                            v2_POI = 15
                        elif x__ > -0.0686 * veh_l[1]:
                            v2_POI = 14
                        elif x__ > -0.284 * veh_l[1]:
                            v2_POI = 13
                        else:
                            v2_POI = 12

                        distance = (veh_l[1] / 2) - x__
                        crash_condition.append([2, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False

            elif y__old < -(veh_w[1] / 2) and y__ > -(veh_w[1] / 2):

                if delta_angle == 270:
                    if num in [2, 3]:
                        if num == 3:
                            if x__ - veh_w[0] < - veh_l[1] / 2:
                                x__2 = (x__ - veh_l[1] / 2) / 2
                            else:
                                x__2 = x__ - veh_w[0] / 2
                            y__1 = x__2 - x__ - veh_w[0] / 2

                        elif num == 2:
                            if x__ + veh_w[0] > veh_l[1] / 2:
                                x__2 = (x__ + veh_l[1] / 2) / 2
                            else:
                                x__2 = x__ + veh_w[0] / 2
                            y__1 = - (x__2 - x__ + veh_w[0] / 2)

                        if y__1 > 0.25 * veh_w[0]:
                            v1_POI = 11
                        elif y__1 > 0.05 * veh_w[0]:
                            v1_POI = 10
                        elif y__1 > -0.05 * veh_w[0]:
                            v1_POI = F
                        elif y__1 > -0.25 * veh_w[0]:
                            v1_POI = 9
                        else:
                            v1_POI = 8

                        if x__2 > 0.147 * veh_l[1]:
                            v2_POI = 4
                        elif x__2 > -0.0686 * veh_l[1]:
                            v2_POI = 5
                        elif x__2 > -0.284 * veh_l[1]:
                            v2_POI = 6
                        else:
                            v2_POI = 7

                        distance = (veh_l[1] / 2) - x__2
                        crash_condition.append([2, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False

                elif delta_angle == 0 or delta_angle == 180:
                    if (num == 3 and 357 < delta_angle_0 < 360) or (num == 0 and 0 < delta_angle_0 < 3) or (
                            num == 1 and 177 < delta_angle_0 < 180) or (num == 2 and 180 < delta_angle_0 < 183):
                        v1_POI = [D, A, B, C][num]
                        if x__ > 0.147 * veh_l[1]:
                            v2_POI = 4
                        elif x__ > -0.0686 * veh_l[1]:
                            v2_POI = 5
                        elif x__ > -0.284 * veh_l[1]:
                            v2_POI = 6
                        else:
                            v2_POI = 7

                        distance = (veh_l[1] / 2) - x__
                        crash_condition.append([2, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False

                elif delta_angle == 90:
                    if num in [0, 1]:
                        if num == 0:
                            if x__ + veh_w[0] > veh_l[1] / 2:
                                x__2 = (x__ + veh_l[1] / 2) / 2
                            else:
                                x__2 = x__ + veh_w[0] / 2
                            y__1 = -(x__2 - x__ - veh_w[0] / 2)

                        elif num == 1:
                            if x__ - veh_w[0] < - veh_l[1] / 2:
                                x__2 = (x__ - veh_l[1] / 2) / 2
                            else:
                                x__2 = x__ - veh_w[0] / 2
                            y__1 = -(x__2 - x__ + veh_w[0] / 2)

                        if y__1 > 0.25 * veh_w[0]:
                            v1_POI = 16
                        elif y__1 > 0.05 * veh_w[0]:
                            v1_POI = 1
                        elif y__1 > -0.05 * veh_w[0]:
                            v1_POI = E
                        elif y__1 > -0.25 * veh_w[0]:
                            v1_POI = 2
                        else:
                            v1_POI = 3

                        if x__2 > 0.147 * veh_l[1]:
                            v2_POI = 4
                        elif x__2 > -0.0686 * veh_l[1]:
                            v2_POI = 5
                        elif x__2 > -0.284 * veh_l[1]:
                            v2_POI = 6
                        else:
                            v2_POI = 7

                        distance = (veh_l[1] / 2) - x__2
                        crash_condition.append([2, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False
                else:
                    if (num == 0 and 90 < delta_angle < 180) or (num == 1 and 0 < delta_angle < 90):
                        flag_error = False
                    else:
                        v1_POI = [D, A, B, C][num]
                        if x__ > 0.147 * veh_l[1]:
                            v2_POI = 4
                        elif x__ > -0.0686 * veh_l[1]:
                            v2_POI = 5
                        elif x__ > -0.284 * veh_l[1]:
                            v2_POI = 6
                        else:
                            v2_POI = 7

                        distance = (veh_l[1] / 2) - x__
                        crash_condition.append([2, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False

            if (veh_l[1] / 2) * 0.6 < np.abs(x__) < (veh_l[1] / 2) and (veh_w[1] / 2) * 0.6 < np.abs(y__) < (veh_w[1] / 2):
                if crash_condition == []:
                    v1_POI = [D, A, B, C][num]
                    if (veh_l[1] / 2) * 0.6 < x__ < (veh_l[1] / 2):
                        if (veh_w[1] / 2) * 0.6 < y__ < (veh_w[1] / 2):
                            v2_POI = D
                        elif - (veh_w[1] / 2) * 0.6 > y__ > - (veh_w[1] / 2):
                            v2_POI = A

                    elif - (veh_l[1] / 2) * 0.6 > x__ > - (veh_l[1] / 2):
                        if (veh_w[1] / 2) * 0.6 < y__ < (veh_w[1] / 2):
                            v2_POI = C
                        elif - (veh_w[1] / 2) * 0.6 > y__ > - (veh_w[1] / 2):
                            v2_POI = B

                    distance = 0.1 * (veh_w[1] / 2)
                    crash_condition.append([1, v1_POI - 1, v2_POI - 1, distance])
                    flag_error = False

            if flag_error:
                print('Detection error!')


    # Calculate the relative angle.
    delta_angle_0 = np.rad2deg(v2_angle - v1_angle)
    delta_angle_0 = delta_angle_0 if delta_angle_0 >= 0 else delta_angle_0 + 360
    delta_angle = ((delta_angle_0 + 2.5) // 5) * 5
    delta_angle = 0 if delta_angle == 360 else delta_angle

    # Detect four vertexes of Vehicle II in turn.
    num = -1
    for (x, y) in [(v2_x1, v2_y1), (v2_x2, v2_y2), (v2_x3, v2_y3), (v2_x4, v2_y4)]:
        num += 1

        # Calculate the relative positions.
        x_ = x - v1_x
        y_ = y - v1_y
        x__ = x_ * np.cos(v1_angle) + y_ * np.sin(v1_angle)
        y__ = y_ * np.cos(v1_angle) - x_ * np.sin(v1_angle)

        # Detect whether a vertex of Vehicle II touches Vehicle I.
        if np.abs(x__) <= (veh_l[0] / 2) and np.abs(y__) <= (veh_w[0] / 2):
            sign1 = 1 if num < 1.5 else -1
            sign2 = 1 if 0.5 < num < 2.5 else -1

            v2_x_old_ = v2_x_old + sign1 * (veh_l[1] / 2) * np.cos(v2_angle_old) + sign2 * (veh_w[1] / 2) * np.sin(
                v2_angle_old)
            v2_y_old_ = v2_y_old + sign1 * (veh_l[1] / 2) * np.sin(v2_angle_old) - sign2 * (veh_w[1] / 2) * np.cos(
                v2_angle_old)

            # Calculate the relative positions at the previous time step.
            x_old = v2_x_old_ - v1_x_old
            y_old = v2_y_old_ - v1_y_old
            x__old = x_old * np.cos(v1_angle_old) + y_old * np.sin(v1_angle_old)
            y__old = y_old * np.cos(v1_angle_old) - x_old * np.sin(v1_angle_old)

            flag_error = True

            # Determine two vehicles' point of impact (POI).
            if x__old > (veh_l[0] / 2) and x__ < (veh_l[0] / 2):

                if delta_angle in [90, 270]:
                    if (num == 1 and delta_angle == 270) or (num == 3 and delta_angle == 90):
                        y__1 = (y__ + veh_w[0] / 2) / 2
                        if y__1 > 0.25 * veh_w[0]:
                            v1_POI = 16
                        elif y__1 > 0.05 * veh_w[0]:
                            v1_POI = 1
                        elif y__1 > -0.05 * veh_w[0]:
                            v1_POI = E
                        elif y__1 > -0.25 * veh_w[0]:
                            v1_POI = 2
                        else:
                            v1_POI = 3

                        if (num == 1 and delta_angle == 270):
                            x__2 = veh_l[1] / 2 - (veh_w[0] / 2 - y__1)
                            if x__2 > 0.147 * veh_l[1]:
                                v2_POI = 4
                            elif x__2 > -0.0686 * veh_l[1]:
                                v2_POI = 5
                            elif x__2 > -0.284 * veh_l[1]:
                                v2_POI = 6
                            else:
                                v2_POI = 7

                        elif (num == 3 and delta_angle == 90):
                            x__2 = - (veh_l[1] / 2 - (veh_w[0] / 2 - y__1))
                            if x__2 > 0.147 * veh_l[1]:
                                v2_POI = 15
                            elif x__2 > -0.0686 * veh_l[1]:
                                v2_POI = 14
                            elif x__2 > -0.284 * veh_l[1]:
                                v2_POI = 13
                            else:
                                v2_POI = 12

                        distance = (veh_l[1] / 2) - x__2
                        crash_condition.append([2, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False

                    if (num == 0 and delta_angle == 90) or (num == 2 and delta_angle == 270):
                        y__1 = (y__ - veh_w[0] / 2) / 2
                        if y__1 > 0.25 * veh_w[0]:
                            v1_POI = 16
                        elif y__1 > 0.05 * veh_w[0]:
                            v1_POI = 1
                        elif y__1 > -0.05 * veh_w[0]:
                            v1_POI = E
                        elif y__1 > -0.25 * veh_w[0]:
                            v1_POI = 2
                        else:
                            v1_POI = 3

                        if (num == 0 and delta_angle == 90):
                            x__2 = veh_l[1] / 2 - (veh_w[0] / 2 + y__1)
                            if x__2 > 0.147 * veh_l[1]:
                                v2_POI = 15
                            elif x__2 > -0.0686 * veh_l[1]:
                                v2_POI = 14
                            elif x__2 > -0.284 * veh_l[1]:
                                v2_POI = 13
                            else:
                                v2_POI = 12

                        elif (num == 2 and delta_angle == 270):
                            x__2 = - (veh_l[1] / 2 - (veh_w[0] / 2 + y__1))
                            if x__2 > 0.147 * veh_l[1]:
                                v2_POI = 4
                            elif x__2 > -0.0686 * veh_l[1]:
                                v2_POI = 5
                            elif x__2 > -0.284 * veh_l[1]:
                                v2_POI = 6
                            else:
                                v2_POI = 7

                        distance = (veh_l[1] / 2) - x__2
                        crash_condition.append([2, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False

                elif delta_angle == 0:
                    if num in [2, 3]:
                        if num == 2:
                            if y__ + veh_w[1] > veh_w[0] / 2:
                                y__1 = (y__ + veh_w[0] / 2) / 2
                                if len(crash_condition) > 0:
                                    continue
                            else:
                                y__1 = y__ + veh_w[1] / 2
                            y__2 = y__1 - y__ - veh_w[1] / 2

                        elif num == 3:
                            if y__ - veh_w[1] < - veh_w[0] / 2:
                                y__1 = (y__ - veh_w[0] / 2) / 2
                                if len(crash_condition) > 0:
                                    continue
                            else:
                                y__1 = y__ - veh_w[1] / 2
                            y__2 = y__1 - y__ + veh_w[1] / 2

                        if y__2 > 0.25 * veh_w[1]:
                            v2_POI = 11
                        elif y__2 > 0.05 * veh_w[1]:
                            v2_POI = 10
                        elif y__2 > -0.05 * veh_w[1]:
                            v2_POI = F
                        elif y__2 > -0.25 * veh_w[1]:
                            v2_POI = 9
                        else:
                            v2_POI = 8

                        if y__1 > 0.25 * veh_w[0]:
                            v1_POI = 16
                        elif y__1 > 0.05 * veh_w[0]:
                            v1_POI = 1
                        elif y__1 > -0.05 * veh_w[0]:
                            v1_POI = E
                        elif y__1 > -0.25 * veh_w[0]:
                            v1_POI = 2
                        else:
                            v1_POI = 3

                        distance = (veh_w[1] / 2) - np.abs(y__2)
                        crash_condition.append([1, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False

                elif delta_angle == 180:
                    if num in [0, 1]:
                        if num == 0:
                            if y__ + veh_w[1] > veh_w[0] / 2:
                                y__1 = (y__ + veh_w[0] / 2) / 2
                                if len(crash_condition) > 0:
                                    continue
                            else:
                                y__1 = y__ + veh_w[1] / 2
                            y__2 = -(y__1 - y__ - veh_w[1] / 2)

                        elif num == 1:
                            if y__ - veh_w[1] < - veh_w[0] / 2:
                                y__1 = (y__ - veh_w[0] / 2) / 2
                                if len(crash_condition) > 0:
                                    continue
                            else:
                                y__1 = y__ - veh_w[1] / 2
                            y__2 = -(y__1 - y__ + veh_w[1] / 2)

                        if y__2 > 0.25 * veh_w[1]:
                            v2_POI = 16
                        elif y__2 > 0.05 * veh_w[1]:
                            v2_POI = 1
                        elif y__2 > -0.05 * veh_w[1]:
                            v2_POI = E
                        elif y__2 > -0.25 * veh_w[1]:
                            v2_POI = 2
                        else:
                            v2_POI = 3

                        if y__1 > 0.25 * veh_w[0]:
                            v1_POI = 16
                        elif y__1 > 0.05 * veh_w[0]:
                            v1_POI = 1
                        elif y__1 > -0.05 * veh_w[0]:
                            v1_POI = E
                        elif y__1 > -0.25 * veh_w[0]:
                            v1_POI = 2
                        else:
                            v1_POI = 3

                        distance = (veh_w[0] / 2) - np.abs(y__1)
                        crash_condition.append([3, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                else:
                    v2_POI = [D, A, B, C][num]
                    if (v2_POI == D and 90 < delta_angle < 180) or (v2_POI == A and 180 < delta_angle < 270) or (
                            v2_POI == B and 270 < delta_angle < 360) or (v2_POI == C and 0 < delta_angle < 90):
                        if y__ > 0.25 * veh_w[0]:
                            v1_POI = 16
                        elif y__ > 0:
                            v1_POI = 1
                        elif y__ > -0.25 * veh_w[0]:
                            v1_POI = 2
                        else:
                            v1_POI = 3

                        distance = (veh_w[0] / 2) - np.abs(y__)
                        crash_condition.append([3, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False

            elif x__old < -(veh_l[0] / 2) and x__ > -(veh_l[0] / 2):

                if delta_angle in [90, 180, 270]:
                    pass
                elif delta_angle == 0:
                    if num == 0:
                        if y__ - veh_w[1] < - veh_w[0] / 2:
                            y__1 = (y__ - veh_w[0] / 2) / 2
                        else:
                            y__1 = y__ - veh_w[1] / 2
                        y__2 = y__1 - y__ + veh_w[1] / 2

                    elif num == 1:
                        if y__ + veh_w[1] > veh_w[0] / 2:
                            y__1 = (y__ + veh_w[0] / 2) / 2
                        else:
                            y__1 = y__ + veh_w[1] / 2
                        y__2 = y__1 - y__ - veh_w[1] / 2

                    if y__2 > 0.25 * veh_w[1]:
                        v2_POI = 16
                    elif y__2 > 0.05 * veh_w[1]:
                        v2_POI = 1
                    elif y__2 > -0.05 * veh_w[1]:
                        v2_POI = E
                    elif y__2 > -0.25 * veh_w[1]:
                        v2_POI = 2
                    else:
                        v2_POI = 3

                    if y__1 > 0.25 * veh_w[0]:
                        v1_POI = 11
                    elif y__1 > 0.05 * veh_w[0]:
                        v1_POI = 10
                    elif y__1 > -0.05 * veh_w[0]:
                        v1_POI = F
                    elif y__1 > -0.25 * veh_w[0]:
                        v1_POI = 9
                    else:
                        v1_POI = 8

                    distance = (veh_w[0] / 2) - np.abs(y__1)
                    crash_condition.append([3, v1_POI - 1, v2_POI - 1, distance])
                    flag_error = False
                else:
                    v2_POI = [D, A, B, C][num]
                    if (v2_POI == D and 270 < delta_angle < 360) or (v2_POI == A and 0 < delta_angle < 90):
                        if y__ > 0.25 * veh_w[0]:
                            v1_POI = 11
                        elif y__ > 0:
                            v1_POI = 10
                        elif y__ > -0.25 * veh_w[0]:
                            v1_POI = 9
                        else:
                            v1_POI = 8

                        distance = (veh_w[0] / 2) - np.abs(y__)
                        crash_condition.append([3, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False

            if y__old > (veh_w[0] / 2) and y__ < (veh_w[0] / 2):
                if delta_angle == 90:
                    if num in [2, 3]:
                        if num == 2:
                            if x__ - veh_w[1] < - veh_l[0] / 2:
                                x__1 = (x__ - veh_l[0] / 2) / 2
                            else:
                                x__1 = x__ - veh_w[1] / 2
                            y__2 = - (x__1 - x__ + veh_w[1] / 2)

                        elif num == 3:
                            if x__ + veh_w[1] > veh_l[0] / 2:
                                x__1 = (x__ + veh_l[0] / 2) / 2
                            else:
                                x__1 = x__ + veh_w[1] / 2
                            y__2 = x__1 - x__ - veh_w[1] / 2

                        if y__2 > 0.25 * veh_w[1]:
                            v2_POI = 11
                        elif y__2 > 0.05 * veh_w[1]:
                            v2_POI = 10
                        elif y__2 > -0.05 * veh_w[1]:
                            v2_POI = F
                        elif y__2 > -0.25 * veh_w[1]:
                            v2_POI = 9
                        else:
                            v2_POI = 8

                        if x__1 > 0.147 * veh_l[0]:
                            v1_POI = 15
                        elif x__1 > -0.0686 * veh_l[0]:
                            v1_POI = 14
                        elif x__1 > -0.284 * veh_l[0]:
                            v1_POI = 13
                        else:
                            v1_POI = 12

                        distance = (veh_l[0] / 2) - x__1
                        crash_condition.append([4, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False

                elif delta_angle == 0 or delta_angle == 180:
                    if (num == 1 and 357 < delta_angle_0 < 360) or (num == 2 and 0 < delta_angle_0 < 3) or (
                            num == 3 and 177 < delta_angle_0 < 180) or (num == 0 and 180 < delta_angle_0 < 183):
                        v2_POI = [D, A, B, C][num]
                        if x__ > 0.147 * veh_l[0]:
                            v1_POI = 15
                        elif x__ > -0.0686 * veh_l[0]:
                            v1_POI = 14
                        elif x__ > -0.284 * veh_l[0]:
                            v1_POI = 13
                        else:
                            v1_POI = 12

                        distance = (veh_l[0] / 2) - x__
                        crash_condition.append([4, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False

                elif delta_angle == 270:
                    if num in [0, 1]:
                        if num == 0:
                            if x__ - veh_w[1] < - veh_l[0] / 2:
                                x__1 = (x__ - veh_l[0] / 2) / 2
                            else:
                                x__1 = x__ - veh_w[1] / 2
                            y__2 = x__1 - x__ + veh_w[1] / 2

                        else:
                            if x__ + veh_w[1] > veh_l[0] / 2:
                                x__1 = (x__ + veh_l[0] / 2) / 2
                            else:
                                x__1 = x__ + veh_w[1] / 2
                            y__2 = x__1 - x__ - veh_w[1] / 2

                        if y__2 > 0.25 * veh_w[1]:
                            v2_POI = 16
                        elif y__2 > 0.05 * veh_w[1]:
                            v2_POI = 1
                        elif y__2 > -0.05 * veh_w[1]:
                            v2_POI = E
                        elif y__2 > -0.25 * veh_w[1]:
                            v2_POI = 2
                        else:
                            v2_POI = 3

                        if x__1 > 0.147 * veh_l[0]:
                            v1_POI = 15
                        elif x__1 > -0.0686 * veh_l[0]:
                            v1_POI = 14
                        elif x__1 > -0.284 * veh_l[0]:
                            v1_POI = 13
                        else:
                            v1_POI = 12

                        distance = (veh_l[0] / 2) - x__1
                        crash_condition.append([4, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False
                else:
                    if (num == 1 and 180 < delta_angle < 270) or (num == 0 and 270 < delta_angle < 360):
                        flag_error = False
                    else:
                        v2_POI = [D, A, B, C][num]
                        if x__ > 0.147 * veh_l[0]:
                            v1_POI = 15
                        elif x__ > -0.0686 * veh_l[0]:
                            v1_POI = 14
                        elif x__ > -0.284 * veh_l[0]:
                            v1_POI = 13
                        else:
                            v1_POI = 12

                        distance = (veh_l[0] / 2) - x__
                        crash_condition.append([4, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False

            elif y__old < -(veh_w[0] / 2) and y__ > -(veh_w[0] / 2):

                if delta_angle == 270:
                    if num in [2, 3]:
                        if num == 3:
                            if x__ - veh_w[1] < - veh_l[0] / 2:
                                x__1 = (x__ - veh_l[0] / 2) / 2
                            else:
                                x__1 = x__ - veh_w[1] / 2
                            y__2 = x__1 - x__ - veh_w[1] / 2

                        elif num == 2:
                            if x__ + veh_w[1] > veh_l[0] / 2:
                                x__1 = (x__ + veh_l[0] / 2) / 2
                            else:
                                x__1 = x__ + veh_w[1] / 2
                            y__2 = - (x__1 - x__ + veh_w[1] / 2)

                        if y__2 > 0.25 * veh_w[1]:
                            v2_POI = 11
                        elif y__2 > 0.05 * veh_w[1]:
                            v2_POI = 10
                        elif y__2 > -0.05 * veh_w[1]:
                            v2_POI = F
                        elif y__2 > -0.25 * veh_w[1]:
                            v2_POI = 9
                        else:
                            v2_POI = 8

                        if x__1 > 0.147 * veh_l[0]:
                            v1_POI = 4
                        elif x__1 > -0.0686 * veh_l[0]:
                            v1_POI = 5
                        elif x__1 > -0.284 * veh_l[0]:
                            v1_POI = 6
                        else:
                            v1_POI = 7

                        distance = (veh_l[0] / 2) - x__1
                        crash_condition.append([4, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False

                elif delta_angle == 0 or delta_angle == 180:
                    if (num == 3 and 357 < delta_angle_0 < 360) or (num == 0 and 0 < delta_angle_0 < 3) or (
                            num == 1 and 177 < delta_angle_0 < 180) or (num == 2 and 180 < delta_angle_0 < 183):
                        v2_POI = [D, A, B, C][num]
                        if x__ > 0.147 * veh_l[0]:
                            v1_POI = 4
                        elif x__ > -0.0686 * veh_l[0]:
                            v1_POI = 5
                        elif x__ > -0.284 * veh_l[0]:
                            v1_POI = 6
                        else:
                            v1_POI = 7

                        distance = (veh_l[0] / 2) - x__
                        crash_condition.append([4, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False

                elif delta_angle == 90:
                    if num in [0, 1]:
                        if num == 0:
                            if x__ + veh_w[1] > veh_l[0] / 2:
                                x__1 = (x__ + veh_l[0] / 2) / 2
                            else:
                                x__1 = x__ + veh_w[1] / 2
                            y__2 = -(x__1 - x__ - veh_w[1] / 2)

                        elif num == 1:
                            if x__ - veh_w[1] < - veh_l[0] / 2:
                                x__1 = (x__ - veh_l[0] / 2) / 2
                            else:
                                x__1 = x__ - veh_w[1] / 2
                            y__2 = -(x__1 - x__ + veh_w[1] / 2)

                        if y__2 > 0.25 * veh_w[1]:
                            v2_POI = 16
                        elif y__2 > 0.05 * veh_w[1]:
                            v2_POI = 1
                        elif y__2 > -0.05 * veh_w[1]:
                            v2_POI = E
                        elif y__2 > -0.25 * veh_w[1]:
                            v2_POI = 2
                        else:
                            v2_POI = 3

                        if x__1 > 0.147 * veh_l[0]:
                            v1_POI = 4
                        elif x__1 > -0.0686 * veh_l[0]:
                            v1_POI = 5
                        elif x__1 > -0.284 * veh_l[0]:
                            v1_POI = 6
                        else:
                            v1_POI = 7

                        distance = (veh_l[0] / 2) - x__1
                        crash_condition.append([4, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False
                    else:
                        flag_error = False
                else:
                    if (num == 0 and 90 < delta_angle < 180) or (num == 1 and 0 < delta_angle < 90):
                        flag_error = False
                    else:
                        v2_POI = [D, A, B, C][num]
                        if x__ > 0.147 * veh_l[0]:
                            v1_POI = 4
                        elif x__ > -0.0686 * veh_l[0]:
                            v1_POI = 5
                        elif x__ > -0.284 * veh_l[0]:
                            v1_POI = 6
                        else:
                            v1_POI = 7

                        distance = (veh_l[0] / 2) - x__
                        crash_condition.append([4, v1_POI - 1, v2_POI - 1, distance])
                        flag_error = False

            if (veh_l[0] / 2) * 0.6 < np.abs(x__) < (veh_l[0] / 2) and (veh_w[0] / 2) * 0.6 < np.abs(y__) < (veh_w[0] / 2):
                if crash_condition == []:
                    v2_POI = [D, A, B, C][num]
                    if (veh_l[0] / 2) * 0.6 < x__ < (veh_l[0] / 2):
                        if (veh_w[0] / 2) * 0.6 < y__ < (veh_w[0] / 2):
                            v1_POI = D
                        elif - (veh_w[0] / 2) * 0.6 > y__ > - (veh_w[0] / 2):
                            v1_POI = A

                    elif - (veh_l[0] / 2) * 0.6 > x__ > - (veh_l[0] / 2):
                        if (veh_w[0] / 2) * 0.6 < y__ < (veh_w[0] / 2):
                            v1_POI = C
                        elif - (veh_w[0] / 2) * 0.6 > y__ > - (veh_w[0] / 2):
                            v1_POI = B

                    distance = 0.1 * (veh_w[0] / 2)
                    crash_condition.append([3, v1_POI - 1, v2_POI - 1, distance])
                    flag_error = False

            if flag_error:
                print('Detection error!')

    if crash_condition == []:
        return False
    else:
        if len(crash_condition) > 1:
            if crash_condition[0][:3] == crash_condition[1][:3] and -0.01 < crash_condition[0][3] - crash_condition[1][
                3] < 0.01:
                crash_condition.pop(1)
            if 15 < crash_condition[0][1] < 20 and 15 < crash_condition[0][2] < 20:
                crash_condition.pop(0)

        return crash_condition
