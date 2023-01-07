import numpy as np
from numba import njit


def euc_dist(k, k_1):
    # calculates the shortest distance between two data points

    n = len(k)
    sum = 0
    for i in range(n):
        sum += (k[i] - k_1[i]) ** 2
    return (sum) ** 0.5


def compute_knots_vector(input_data, degree):
    head_list = [0] * degree
    tail_list = [1] * degree
    knots = head_list + input_data + tail_list
    return knots

def parameterization(points, degree, u):
    if u == True:
        param = []
        for i in points:
            point_lenght = len(i)
            parameters = [0] * point_lenght

            for j in range(1, point_lenght):
                a = 0
                b = 0
                for k in range(1, j + 1):
                    a += euc_dist(i[k], i[k - 1])
                for k in range(1, point_lenght):
                    b += euc_dist(i[k], i[k - 1])
                parameters[j] = a / b

            parameters[-1] = 1
            param.append(parameters)
        params = [sum(z) for z in zip(*param)]
        param[:] = [x / len(param) for x in params]
        # print(param)
        knots = compute_knots_vector(param, degree)

    else:

        k = []
        for i in zip(*points):
            k.append(i)
        points_v = [list(x) for x in k]

        param = []
        for i in points_v:
            point_lenght = len(i)
            parameters = [0] * point_lenght

            for j in range(1, point_lenght):
                a = 0
                b = 0
                for k in range(1, j + 1):
                    a += euc_dist(i[k], i[k - 1])
                for k in range(1, point_lenght):
                    b += euc_dist(i[k], i[k - 1])
                parameters[j] = a / b

            parameters[-1] = 1
            param.append(parameters)
        params = [sum(z) for z in zip(*param)]
        param[:] = [x / len(param) for x in params]
        knots = compute_knots_vector(param, degree)

    return param, knots



def basis_function(parameters, knots, i, j):
    N_value = 0.

    if knots[j] <= parameters[i] <= knots[j + 1] and (parameters[i] != knots[j] or parameters[i] != knots[j + 1]):
        try:
            N_value = (parameters[i] - knots[j]) ** 3 / (
                        (knots[j + 1] - knots[j]) * (knots[j + 2] - knots[j]) * (knots[j + 3] - knots[j]))
        except ZeroDivisionError:
            N_value = 0.

    elif knots[j + 1] <= parameters[i] < knots[j + 2]:
        try:
            N_value = ((parameters[i] - knots[j]) ** 2 * (knots[j + 2] - parameters[i])) / (
                        (knots[j + 2] - knots[j + 1]) * (knots[j + 3] - knots[j]) * (knots[j + 2] - knots[j])) + \
                      ((knots[j + 3] - parameters[i]) * (parameters[i] - knots[j]) * (parameters[i] - knots[j + 1])) / (
                                  (knots[j + 2] - knots[j + 1]) * (knots[j + 3] - knots[j + 1]) * (
                                      knots[j + 3] - knots[j])) + \
                      ((knots[j + 4] - parameters[i]) * ((parameters[i] - knots[j + 1]) ** 2)) / (
                                  (knots[j + 2] - knots[j + 1]) * (knots[j + 4] - knots[j + 1]) * (
                                      knots[j + 3] - knots[j + 1]))
        except ZeroDivisionError:
            N_value = 0.

    elif knots[j + 2] <= parameters[i] < knots[j + 3]:
        try:
            N_value = ((parameters[i] - knots[j]) * (knots[j + 3] - parameters[i]) ** 2) / (
                        (knots[j + 3] - knots[j + 2]) * (knots[j + 3] - knots[j + 1]) * (knots[j + 3] - knots[j])) + \
                      ((knots[j + 4] - parameters[i]) * (knots[j + 3] - parameters[i]) * (
                                  parameters[i] - knots[j + 1])) / (
                                  (knots[j + 3] - knots[j + 2]) * (knots[j + 4] - knots[j + 1]) * (
                                      knots[j + 3] - knots[j + 1])) + \
                      ((knots[j + 4] - parameters[i]) ** 2 * (parameters[i] - knots[j + 2])) / (
                                  (knots[j + 3] - knots[j + 2]) * (knots[j + 4] - knots[j + 2]) * (
                                      knots[j + 4] - knots[j + 1]))
        except ZeroDivisionError:
            N_value = 0.

    elif knots[j + 3] <= parameters[i] <= knots[j + 4] and (
            parameters[i] != knots[j + 3] or parameters[i] != knots[j + 4]):
        try:
            N_value = (knots[j + 4] - parameters[i]) ** 3 / (
                        (knots[j + 4] - knots[j + 3]) * (knots[j + 4] - knots[j + 2]) * (knots[j + 4] - knots[j + 1]))
        except ZeroDivisionError:
            N_value = 0.

    return N_value


def boundary_function(parameters, knots, i, j):
    N_value = 0.

    if knots[j] <= parameters[i] <= knots[j + 1] and (parameters[i] != knots[j] or parameters[i] != knots[j + 1]):
        try:
            N_value = 6 * (parameters[i] - knots[j]) / (
                        (knots[j + 1] - knots[j]) * (knots[j + 2] - knots[j]) * (knots[j + 3] - knots[j]))
        except ZeroDivisionError:
            N_value = 0.

    elif knots[j + 1] <= parameters[i] <= knots[j + 2] and (
            parameters[i] != knots[j + 1] or parameters[i] != knots[j + 2]):
        try:
            N_value = (2 * (knots[j + 2] - parameters[i]) - 4 * (parameters[i] - knots[j])) / (
                        (knots[j + 2] - knots[j + 1]) * (knots[j + 3] - knots[j]) * (knots[j + 2] - knots[j])) + \
                      (2 * knots[j] - 6 * parameters[i] + 2 * knots[j + 1] + 2 * knots[j + 3]) / (
                                  (knots[j + 2] - knots[j + 1]) * (knots[j + 3] - knots[j + 1]) * (
                                      knots[j + 3] - knots[j])) + \
                      (4 * knots[j + 1] - 6 * parameters[i] + 2 * knots[j + 4]) / (
                                  (knots[j + 2] - knots[j + 1]) * (knots[j + 4] - knots[j + 1]) * (
                                      knots[j + 3] - knots[j + 1]))
        except ZeroDivisionError:
            N_value = 0.

    elif knots[j + 2] <= parameters[i] <= knots[j + 3] and (
            parameters[i] != knots[j + 2] or parameters[i] != knots[j + 3]):
        try:
            N_value = (6 * parameters[i] - 2 * knots[j] - 4 * knots[j + 3]) / (
                        (knots[j + 3] - knots[j + 2]) * (knots[j + 3] - knots[j + 1]) * (knots[j + 3] - knots[j])) + \
                      (6 * parameters[i] - 2 * knots[j + 1] - 2 * knots[j + 3] - 2 * knots[j + 4]) / (
                                  (knots[j + 3] - knots[j + 2]) * (knots[j + 4] - knots[j + 1]) * (
                                      knots[j + 3] - knots[j + 1])) + \
                      (6 * parameters[i] - 2 * knots[j + 2] - 4 * knots[j + 4]) / (
                                  (knots[j + 3] - knots[j + 2]) * (knots[j + 4] - knots[j + 2]) * (
                                      knots[j + 4] - knots[j + 1]))
        except ZeroDivisionError:
            N_value = 0.

    elif knots[j + 3] <= parameters[i] <= knots[j + 4] and (
            parameters[i] != knots[j + 3] or parameters[i] != knots[j + 4]):
        try:
            N_value = 6 * (knots[j + 4] - parameters[i]) / (
                        (knots[j + 4] - knots[j + 3]) * (knots[j + 4] - knots[j + 2]) * (knots[j + 4] - knots[j + 1]))
        except ZeroDivisionError:
            N_value = 0.

    return N_value


def spline_matrix(parameters, knots):
    # Cubic B-spline basis and end function

    lenght_ = len(parameters) - 1
    ctrl_number = lenght_ + 3  # control no #3
    spline_matrix = np.zeros((ctrl_number, ctrl_number), dtype=float).tolist()

    # Basis Function
    for i in range(lenght_ + 1):
        for j in range(lenght_ + 3):
            spline_matrix[i + 1][j] = basis_function(parameters, knots, i, j)

    # End point Function: this is to fix/tie the first and last values to the spline
    for i in range(2):
        for j in range(lenght_ + 3):
            k = i * lenght_
            spline_matrix[i * (lenght_ + 2)][j] = boundary_function(parameters, knots, k, j)

    return spline_matrix





def all_ctrl(basis_matrix_u, basis_matrix_v, points):
    cp_u = []
    m = len(points)
    for i in range(m):
        control_points_u = TDMAsolver(basis_matrix_u, points[i])
        cp_u.append(control_points_u)

    cp_v = []
    n = len(cp_u[0])
    for i in range(n):
        data = [dat[i] for dat in cp_u]
        control_points_v = TDMAsolver(basis_matrix_v, data)
        cp_v.append(control_points_v)

    data = [item for all_sublist in cp_v for item in all_sublist]
    dat = np.array(data)
    return cp_v, dat


def TDMA(a, b, c, d):
    # Tri-diagonal matrix solver of a system of linear equations

    lenght_ = len(d)
    l_diagonals, m_diagonals, u_diagonals, x_y = map(list, (a, b, c, d))
    for i in range(1, lenght_):
        m = l_diagonals[i - 1]
        n = m_diagonals[i - 1]
        mc = m / n

        m_diagonals[i] = m_diagonals[i] - mc * u_diagonals[i - 1]
        x_y[i] = x_y[i] - mc * x_y[i - 1]

    xc = m_diagonals
    xc[-1] = x_y[-1] / m_diagonals[-1]

    for j in range(lenght_ - 2, -1, -1):
        xc[j] = (x_y[j] - u_diagonals[j] * xc[j + 1]) / m_diagonals[j]

    return xc


def TDMAsolver(basis_matrix, data_point):
    # Tri-diagonal matrix solver of a system of linear equations
    basis_mat = basis_matrix.copy()
    data_points = data_point.copy()

    control_points = []
    n = len(basis_mat[0])
    d = [(0, 0, 0)]
    data_points = d + data_points + d

    x = [i[0] for i in data_points]
    y = [j[1] for j in data_points]
    z = [k[2] for k in data_points]

    basis_mat[0], basis_mat[1] = basis_mat[1], basis_mat[0]
    basis_mat[n - 2], basis_mat[n - 1] = basis_mat[n - 1], basis_mat[n - 2]

    x[0], x[1] = x[1], x[0]
    x[n - 2], x[n - 1] = x[n - 1], x[n - 2]
    y[0], y[1] = y[1], y[0]
    y[n - 2], y[n - 1] = y[n - 1], y[n - 2]
    z[0], z[1] = z[1], z[0]
    z[n - 2], z[n - 1] = z[n - 1], z[n - 2]

    # extract diagonal
    lower_diagonals = tuple([basis_mat[i + 1][i] for i in range(n - 1)])
    main_diagonals = tuple([basis_mat[i][i] for i in range(n)])
    upper_diagonals = tuple([basis_mat[i][i + 1] for i in range(n - 1)])

    x = tuple(x)
    y = tuple(y)
    z = tuple(z)

    x_control = TDMA(lower_diagonals, main_diagonals, upper_diagonals, x)
    y_control = TDMA(lower_diagonals, main_diagonals, upper_diagonals, y)
    z_control = TDMA(lower_diagonals, main_diagonals, upper_diagonals, z)

    for i in range(n):
        control_points.append([x_control[i], y_control[i], z_control[i]])

    return control_points



@njit()
def CDB_recursion(x, degree, i, knots_list):
    if degree == 0:
        return 1.0 if knots_list[i] <= x < knots_list[i + 1] else 0.0
    if knots_list[i + degree] == knots_list[i]:
        c1 = 0.0
    else:
        c1 = (x - knots_list[i]) / (knots_list[i + degree] - knots_list[i]) * CDB_recursion(x, degree - 1, i, knots_list)

    if knots_list[i + degree + 1] == knots_list[i + 1]:
        c2 = 0.0
    else:
        c2 = (knots_list[i + degree + 1] - x) / (knots_list[i + degree + 1] - knots_list[i + 1]) * CDB_recursion(x, degree - 1, i + 1, knots_list)

    y = c1 + c2

    return y

@njit()
def b_spline(x, knots_list, control_points, degree):
    lenght_ = len(knots_list) - degree - 1
    assert (lenght_ >= degree + 1) and (len(control_points) >= lenght_)
    return sum([control_points[i] * CDB_recursion(x, degree, i, knots_list) for i in range(lenght_)])


