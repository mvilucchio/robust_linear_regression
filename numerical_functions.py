# from cv2 import integral
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from tqdm.auto import tqdm
from scipy.integrate import dblquad, quad
import fixed_point_equations as fpe
import numba as nb

MULT_INTEGRAL = 5
A_HUBER = 1.0


@nb.njit(error_model="numpy", fastmath=True)
def ZoutBayes(y, omega, V, delta):
    return np.exp(-((y - omega) ** 2) / (2 * (V + delta))) / np.sqrt(
        2 * np.pi * (V + delta)
    )


@nb.njit(error_model="numpy", fastmath=True)
def foutBayes(y, omega, V, delta):
    return (y - omega) / (V + delta)


@nb.njit(error_model="numpy", fastmath=True)
def foutL2(y, omega, V):
    return (y - omega) / (1 + V)


@nb.njit(error_model="numpy", fastmath=True)
def DfoutL2(y, omega, V):
    return -1.0 / (1 + V)


@nb.njit(error_model="numpy", fastmath=True)
def foutL1(y, omega, V):
    return (y - omega + np.sign(omega - y) * np.maximum(np.abs(omega - y) - V, 0.0)) / V


@nb.njit(error_model="numpy", fastmath=True)
def DfoutL1(y, omega, V):
    if np.abs(omega - y) > V:
        return 0.0
    else:
        return -1.0 / V


@nb.njit(error_model="numpy", fastmath=True)
def foutHuber(y, omega, V, a=A_HUBER):
    if a + a * V + omega < y:
        return a
    elif np.abs(y - omega) <= a + a * V:
        return (y - omega) / (1 + V)
    elif omega > a + a * V + y:
        return -a
    else:
        return 0.0


@nb.njit(error_model="numpy", fastmath=True)
def DfoutHuber(y, omega, V, a=A_HUBER):
    if (y < omega and a + a * V + y < omega) or (a + a * V + omega < y):
        return 0.0
    else:
        return -1.0 / (1 + V)


def find_integration_borders(
    fun, scale1, scale2, mult=MULT_INTEGRAL, tol=1e-6, n_points=300
):
    borders = [[-mult * scale1, mult * scale1], [-mult * scale2, mult * scale2]]

    for idx, ax in enumerate(borders):
        for jdx, border in enumerate(ax):

            while True:
                if idx == 0:
                    max_val = np.max(
                        [
                            fun(borders[idx][jdx], pt)
                            for pt in np.linspace(
                                borders[1 if idx == 0 else 0][0],
                                borders[1 if idx == 0 else 0][1],
                                n_points,
                            )
                        ]
                    )
                else:
                    max_val = np.max(
                        [
                            fun(pt, borders[idx][jdx])
                            for pt in np.linspace(
                                borders[1 if idx == 0 else 0][0],
                                borders[1 if idx == 0 else 0][1],
                                n_points,
                            )
                        ]
                    )
                if max_val > tol:
                    borders[idx][jdx] = borders[idx][jdx] + (
                        -1.0 if jdx == 0 else 1.0
                    ) * (scale1 if idx == 0 else scale2)
                else:
                    break

    for ax in borders:
        ax[0] = -np.max(np.abs(ax))
        ax[1] = np.max(np.abs(ax))

    max_val = np.max([borders[0][1], borders[1][1]])

    borders = [[-max_val, max_val], [-max_val, max_val]]

    return borders


# -----------------
@nb.njit(error_model="numpy", fastmath=True)
def q_integral_BO(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes(y, np.sqrt(q) * xi, 1 - q, delta)
        * (foutBayes(y, np.sqrt(q) * xi, 1 - q, delta) ** 2)
    )


# -----------------
@nb.njit(error_model="numpy", fastmath=True)
def m_integral_L2(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * foutBayes(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * foutL2(y, np.sqrt(q) * xi, sigma)
    )


@nb.njit(error_model="numpy", fastmath=True)
def q_integral_L2(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * (foutL2(y, np.sqrt(q) * xi, sigma) ** 2)
    )


@nb.njit(error_model="numpy", fastmath=True)
def sigma_integral_L2(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * DfoutL2(y, np.sqrt(q) * xi, sigma)
    )


# ---


@nb.njit(error_model="numpy", fastmath=True)
def m_integral_L1(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * foutBayes(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * foutL1(y, np.sqrt(q) * xi, sigma)
    )


@nb.njit(error_model="numpy", fastmath=True)
def q_integral_L1(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * (foutL1(y, np.sqrt(q) * xi, sigma) ** 2)
    )


@nb.njit(error_model="numpy", fastmath=True)
def sigma_integral_L1(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * DfoutL1(y, np.sqrt(q) * xi, sigma)
    )


# ---


@nb.njit(error_model="numpy", fastmath=True)
def m_integral_Huber(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * foutBayes(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * foutHuber(y, np.sqrt(q) * xi, sigma)
    )


@nb.njit(error_model="numpy", fastmath=True)
def q_integral_Huber(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * (foutHuber(y, np.sqrt(q) * xi, sigma) ** 2)
    )


@nb.njit(error_model="numpy", fastmath=True)
def sigma_integral_Huber(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * DfoutHuber(y, np.sqrt(q) * xi, sigma)
    )


# -------------------


def q_hat_equation_BO(m, q, sigma, delta):
    borders = find_integration_borders(
        lambda y, xi: q_integral_BO(y, xi, q, m, sigma, delta), np.sqrt((1 + delta)), 1.0,
    )
    return dblquad(
        q_integral_BO,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        borders[1][1],
        args=(q, m, sigma, delta),
    )[0]


def m_hat_equation_L2(m, q, sigma, delta):
    borders = find_integration_borders(
        lambda y, xi: m_integral_L2(y, xi, q, m, sigma, delta), np.sqrt((1 + delta)), 1.0,
    )
    return dblquad(
        m_integral_L2,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        borders[1][1],
        args=(q, m, sigma, delta),
    )[0]


def q_hat_equation_L2(m, q, sigma, delta):
    borders = find_integration_borders(
        lambda y, xi: q_integral_L2(y, xi, q, m, sigma, delta), np.sqrt((1 + delta)), 1.0,
    )
    return dblquad(
        q_integral_L2,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        borders[1][
            1
        ],  # lambda xi: np.sqrt(eta) * xi - size_y, lambda xi: np.sqrt(eta) * xi + size_y
        args=(q, m, sigma, delta),
    )[0]


def sigma_hat_equation_L2(m, q, sigma, delta):
    borders = find_integration_borders(
        lambda y, xi: sigma_integral_L2(y, xi, q, m, sigma, delta),
        np.sqrt((1 + delta)),
        1.0,
    )
    return dblquad(
        sigma_integral_L2,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        borders[1][1],
        args=(q, m, sigma, delta),
    )[0]


def border_plus_L1(xi, m, q, sigma, delta):
    return


def border_minus_L1(xi, m, q, sigma, delta):
    return


def border_plus(xi, m, q, sigma, delta, a=A_HUBER):
    return np.sqrt(q) / a * xi + (sigma + 1)


def border_minus(xi, m, q, sigma, delta, a=A_HUBER):
    return np.sqrt(q) / a * xi - (sigma + 1)


def test_fun_upper(y, m, q, sigma, delta, a=A_HUBER):
    return a / np.sqrt(q) * (-(sigma + 1) + y)


def test_fun_down(y, m, q, sigma, delta, a=A_HUBER):
    return a / np.sqrt(q) * ((sigma + 1) + y)


def integral_fpe(
    integral_form, border_fun_plus, border_fun_minus, test_function, m, q, sigma, delta
):
    borders = find_integration_borders(
        lambda y, xi: integral_form(y, xi, q, m, sigma, delta), np.sqrt(1 + delta), 1.0
    )

    [_, max_val], _ = borders

    xi_test = test_function(max_val, m, q, sigma, delta)
    xi_test_2 = test_function(-max_val, m, q, sigma, delta)
    # print("xi_test : {} xi_test_2 : {} max_val : {}".format(
    #     xi_test, xi_test_2, max_val))

    if xi_test > max_val:
        #  print("case 1")
        domain_xi = [[-max_val, max_val]] * 3
        domain_y = [
            [lambda xi: border_fun_plus(xi, m, q, sigma, delta), lambda xi: max_val],
            [
                lambda xi: border_fun_minus(xi, m, q, sigma, delta),
                lambda xi: border_fun_plus(xi, m, q, sigma, delta),
            ],
            [lambda xi: -max_val, lambda xi: border_fun_minus(xi, m, q, sigma, delta)],
        ]
    elif xi_test >= 0:
        xi_test_2 = test_function(-max_val, m, q, sigma, delta)
        if xi_test_2 < -max_val:
            #  print("case 2.A")
            domain_xi = [
                [-max_val, xi_test],
                [-xi_test, max_val],
                [-max_val, -xi_test],
                [xi_test, max_val],
                [-xi_test, xi_test],
            ]
            domain_y = [
                [lambda xi: border_fun_plus(xi, m, q, sigma, delta), lambda xi: max_val],
                [
                    lambda xi: -max_val,
                    lambda xi: border_fun_minus(xi, m, q, sigma, delta),
                ],
                [lambda xi: -max_val, lambda xi: border_fun_plus(xi, m, q, sigma, delta)],
                [lambda xi: border_fun_minus(xi, m, q, sigma, delta), lambda xi: max_val],
                [
                    lambda xi: border_fun_minus(xi, m, q, sigma, delta),
                    lambda xi: border_fun_plus(xi, m, q, sigma, delta),
                ],
            ]
        else:
            # print("case 2.B")
            domain_xi = [
                [xi_test_2, xi_test],
                [-xi_test, -xi_test_2],
                [xi_test_2, -xi_test],
                [xi_test, -xi_test_2],
                [-xi_test, xi_test],
                [-max_val, xi_test_2],
                [-xi_test_2, max_val],
            ]
            domain_y = [
                [lambda xi: border_fun_plus(xi, m, q, sigma, delta), lambda xi: max_val],
                [
                    lambda xi: -max_val,
                    lambda xi: border_fun_minus(xi, m, q, sigma, delta),
                ],
                [lambda xi: -max_val, lambda xi: border_fun_plus(xi, m, q, sigma, delta)],
                [lambda xi: border_fun_minus(xi, m, q, sigma, delta), lambda xi: max_val],
                [
                    lambda xi: border_fun_minus(xi, m, q, sigma, delta),
                    lambda xi: border_fun_plus(xi, m, q, sigma, delta),
                ],
                [lambda xi: -max_val, lambda xi: max_val],
                [lambda xi: -max_val, lambda xi: max_val],
            ]
    elif xi_test > -max_val:
        xi_test_2 = test_function(-max_val, m, q, sigma, delta)
        if xi_test_2 < -max_val:
            # print("case 3.A")
            domain_xi = [
                [-max_val, xi_test],
                [-xi_test, max_val],
                [-max_val, xi_test],
                [-xi_test, max_val],
                [xi_test, -xi_test],
            ]
            domain_y = [
                [lambda xi: border_fun_plus(xi, m, q, sigma, delta), lambda xi: max_val],
                [
                    lambda xi: -max_val,
                    lambda xi: border_fun_minus(xi, m, q, sigma, delta),
                ],
                [lambda xi: -max_val, lambda xi: border_fun_plus(xi, m, q, sigma, delta)],
                [lambda xi: border_fun_minus(xi, m, q, sigma, delta), lambda xi: max_val],
                [lambda xi: -max_val, lambda xi: max_val],
            ]
        else:
            # print("case 3.B")
            domain_xi = [
                [xi_test_2, xi_test],
                [-xi_test, -xi_test_2],
                [xi_test_2, xi_test],
                [-xi_test, -xi_test_2],
                [xi_test, -xi_test],
                [-max_val, xi_test_2],
                [-xi_test_2, max_val],
            ]
            domain_y = [
                [lambda xi: border_fun_plus(xi, m, q, sigma, delta), lambda xi: max_val],
                [
                    lambda xi: -max_val,
                    lambda xi: border_fun_minus(xi, m, q, sigma, delta),
                ],
                [lambda xi: -max_val, lambda xi: border_fun_plus(xi, m, q, sigma, delta)],
                [lambda xi: border_fun_minus(xi, m, q, sigma, delta), lambda xi: max_val],
                [lambda xi: -max_val, lambda xi: max_val],
                [lambda xi: -max_val, lambda xi: max_val],
                [lambda xi: -max_val, lambda xi: max_val],
            ]
    else:
        # print("case 4")
        domain_xi = [[-max_val, max_val]]
        domain_y = [[lambda xi: -max_val, lambda xi: max_val]]

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            integral_form,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta),
        )[0]
    return integral_value


def m_hat_equation_L1(m, q, sigma, delta):
    borders = find_integration_borders(
        lambda y, xi: m_integral_L1(y, xi, q, m, sigma, delta), np.sqrt((1 + delta)), 1.0
    )
    first_integral = dblquad(
        m_integral_L1,
        borders[0][0],
        borders[0][1],
        lambda xi: np.sqrt(q) * xi + sigma,
        borders[1][1],
        args=(q, m, sigma, delta),
    )[0]
    second_integral = dblquad(
        m_integral_L1,
        borders[0][0],
        borders[0][1],
        lambda xi: np.sqrt(q) * xi - sigma,
        lambda xi: np.sqrt(q) * xi + sigma,
        args=(q, m, sigma, delta),
    )[0]
    third_integral = dblquad(
        m_integral_L1,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        lambda xi: np.sqrt(q) * xi - sigma,
        args=(q, m, sigma, delta),
    )[0]
    return first_integral + second_integral + third_integral


def q_hat_equation_L1(m, q, sigma, delta):
    borders = find_integration_borders(
        lambda y, xi: q_integral_L1(y, xi, q, m, sigma, delta), np.sqrt((1 + delta)), 1.0,
    )
    first_integral = dblquad(
        q_integral_L1,
        borders[0][0],
        borders[0][1],
        lambda xi: np.sqrt(q) * xi + sigma,
        borders[1][1],
        args=(q, m, sigma, delta),
    )[0]
    second_integral = dblquad(
        q_integral_L1,
        borders[0][0],
        borders[0][1],
        lambda xi: np.sqrt(q) * xi - sigma,
        lambda xi: np.sqrt(q) * xi + sigma,
        args=(q, m, sigma, delta),
    )[0]
    third_integral = dblquad(
        q_integral_L1,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        lambda xi: np.sqrt(q) * xi - sigma,
        args=(q, m, sigma, delta),
    )[0]
    return first_integral + second_integral + third_integral


def sigma_hat_equation_L1(m, q, sigma, delta):
    borders = find_integration_borders(
        lambda y, xi: sigma_integral_L1(y, xi, q, m, sigma, delta),
        np.sqrt((1 + delta)),
        1.0,
    )
    first_integral = dblquad(
        sigma_integral_L1,
        borders[0][0],
        borders[0][1],
        lambda xi: np.sqrt(q) * xi + sigma,
        borders[1][1],
        args=(q, m, sigma, delta),
    )[0]
    second_integral = dblquad(
        sigma_integral_L1,
        borders[0][0],
        borders[0][1],
        lambda xi: np.sqrt(q) * xi - sigma,
        lambda xi: np.sqrt(q) * xi + sigma,
        args=(q, m, sigma, delta),
    )[0]
    third_integral = dblquad(
        sigma_integral_L1,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        lambda xi: np.sqrt(q) * xi - sigma,
        args=(q, m, sigma, delta),
    )[0]
    return first_integral + second_integral + third_integral


def m_hat_equation_Huber(m, q, sigma, delta, a=A_HUBER):
    borders = find_integration_borders(
        lambda y, xi: m_integral_Huber(y, xi, q, m, sigma, delta),
        np.sqrt((1 + delta)),
        1.0,
    )

    test_value = (np.sqrt(q) * borders[0][1] + a * sigma + a) / a
    if test_value > borders[0][1]:
        domain_xi = [borders[0]] * 3
        domain_y = [
            [lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a, lambda xi: borders[1][1],],
            [
                lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a,
                lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a,
            ],
            [lambda xi: borders[1][0], lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a,],
        ]
    elif test_value >= 0:
        domain_xi = [
            [borders[0][0], test_value],
            [-test_value, borders[0][1]],
            [borders[0][0], -test_value],
            [test_value, borders[0][1]],
            [-test_value, test_value],
        ]
        domain_y = [
            [lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a, lambda xi: borders[1][1],],
            [lambda xi: borders[1][0], lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a,],
            [lambda xi: borders[1][0], lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a,],
            [lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a, lambda xi: borders[1][1],],
            [
                lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a,
                lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a,
            ],
        ]
    else:
        domain_xi = [
            [borders[0][0], test_value],
            [-test_value, borders[0][1]],
            [borders[0][0], test_value],
            [-test_value, borders[0][1]],
            [test_value, -test_value],
        ]
        domain_y = [
            [lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a, lambda xi: borders[1][1],],
            [lambda xi: borders[1][0], lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a,],
            [lambda xi: borders[1][0], lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a,],
            [lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a, lambda xi: borders[1][1],],
            [lambda xi: borders[1][0], lambda xi: borders[1][1],],
        ]

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            m_integral_Huber,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta),
        )[0]

    # borders[1][0] = np.min(
    #     [borders[1][0], (np.sqrt(q) * borders[0][0] - a * sigma - a) / a]
    # )
    # borders[1][1] = np.max(
    #     [borders[1][1], (np.sqrt(q) * borders[0][1] + a * sigma + a) / a]
    # )

    # borders[1][0] = -np.max(np.abs(borders[1]))
    # borders[1][1] = np.max(np.abs(borders[1]))

    # first_integral = dblquad(
    #     m_integral_Huber,
    #     borders[0][0],
    #     borders[0][1],
    #     lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a,
    #     borders[1][1],
    #     args=(q, m, sigma, delta),
    # )[0]
    # second_integral = dblquad(
    #     m_integral_Huber,
    #     borders[0][0],
    #     borders[0][1],
    #     lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a,
    #     lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a,
    #     args=(q, m, sigma, delta),
    # )[0]
    # third_integral = dblquad(
    #     m_integral_Huber,
    #     borders[0][0],
    #     borders[0][1],
    #     borders[1][0],
    #     lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a,
    #     args=(q, m, sigma, delta),
    # )[0]
    return integral_value


def q_hat_equation_Huber(m, q, sigma, delta, a=A_HUBER):
    borders = find_integration_borders(
        lambda y, xi: q_integral_Huber(y, xi, q, m, sigma, delta),
        np.sqrt((1 + delta)),
        1.0,
    )

    test_value = (np.sqrt(q) * borders[0][1] + a * sigma + a) / a
    if test_value > borders[0][1]:
        domain_xi = [borders[0]] * 3
        domain_y = [
            [lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a, lambda xi: borders[1][1],],
            [
                lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a,
                lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a,
            ],
            [lambda xi: borders[1][0], lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a,],
        ]
    elif test_value >= 0:
        domain_xi = [
            [borders[0][0], test_value],
            [-test_value, borders[0][1]],
            [borders[0][0], -test_value],
            [test_value, borders[0][1]],
            [-test_value, test_value],
        ]
        domain_y = [
            [lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a, lambda xi: borders[1][1],],
            [lambda xi: borders[1][0], lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a,],
            [lambda xi: borders[1][0], lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a,],
            [lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a, lambda xi: borders[1][1],],
            [
                lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a,
                lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a,
            ],
        ]
    else:
        domain_xi = [
            [borders[0][0], test_value],
            [-test_value, borders[0][1]],
            [borders[0][0], test_value],
            [-test_value, borders[0][1]],
            [test_value, -test_value],
        ]
        domain_y = [
            [lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a, lambda xi: borders[1][1],],
            [lambda xi: borders[1][0], lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a,],
            [lambda xi: borders[1][0], lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a,],
            [lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a, lambda xi: borders[1][1],],
            [lambda xi: borders[1][0], lambda xi: borders[1][1],],
        ]

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_Huber,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta),
        )[0]
    return integral_value


def sigma_hat_equation_Huber(m, q, sigma, delta, a=A_HUBER):
    borders = find_integration_borders(
        lambda y, xi: sigma_integral_Huber(y, xi, q, m, sigma, delta),
        np.sqrt((1 + delta)),
        1.0,
    )

    test_value = (np.sqrt(q) * borders[0][1] + a * sigma + a) / a
    if test_value > borders[0][1]:
        domain_xi = [borders[0]] * 3
        domain_y = [
            [lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a, lambda xi: borders[1][1],],
            [
                lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a,
                lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a,
            ],
            [lambda xi: borders[1][0], lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a,],
        ]
    elif test_value >= 0:
        domain_xi = [
            [borders[0][0], test_value],
            [-test_value, borders[0][1]],
            [borders[0][0], -test_value],
            [test_value, borders[0][1]],
            [-test_value, test_value],
        ]
        domain_y = [
            [lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a, lambda xi: borders[1][1],],
            [lambda xi: borders[1][0], lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a,],
            [lambda xi: borders[1][0], lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a,],
            [lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a, lambda xi: borders[1][1],],
            [
                lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a,
                lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a,
            ],
        ]
    else:
        domain_xi = [
            [borders[0][0], test_value],
            [-test_value, borders[0][1]],
            [borders[0][0], test_value],
            [-test_value, borders[0][1]],
            [test_value, -test_value],
        ]
        domain_y = [
            [lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a, lambda xi: borders[1][1],],
            [lambda xi: borders[1][0], lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a,],
            [lambda xi: borders[1][0], lambda xi: (np.sqrt(q) * xi + a * sigma + a) / a,],
            [lambda xi: (np.sqrt(q) * xi - a * sigma - a) / a, lambda xi: borders[1][1],],
            [lambda xi: borders[1][0], lambda xi: borders[1][1],],
        ]

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            sigma_integral_Huber,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta),
        )[0]
    return integral_value


# ---------------------------


def state_equations_convergence(
    var_func,
    var_hat_func,
    delta=0.1,
    lamb=0.1,
    alpha=0.5,
    init=(0.5, 0.4, 1),
    verbose=False,
):
    m, q, sigma = init[0], init[1], init[2]
    err = 1.0
    blend = 0.6
    iter = 0
    while err > 1e-6:
        m_hat, q_hat, sigma_hat = var_hat_func(m, q, sigma, alpha, delta)

        temp_m = m
        temp_q = q
        temp_sigma = sigma

        m, q, sigma = var_func(m_hat, q_hat, sigma_hat, alpha, delta, lamb)

        err = np.max(np.abs([temp_m - m, temp_q - q, temp_sigma - sigma]))

        m = blend * m + (1 - blend) * temp_m
        q = blend * q + (1 - blend) * temp_q
        sigma = blend * sigma + (1 - blend) * temp_sigma
        if verbose:
            print(f"i = {iter} m = {m}, q = {q}, sigma = {sigma}, eta = {m**2/q}")
        iter += 1
    return m, q, sigma


if __name__ == "__main__":
    # test the convergence
    alpha = 7.4
    deltas = [1.0]
    lambdas = [1.0]

    for idx, l in enumerate(tqdm(lambdas, desc="lambda", leave=False)):
        for jdx, delta in enumerate(tqdm(deltas, desc="delta", leave=False)):
            i = idx * len(deltas) + jdx

            while True:
                m = np.random.random()
                q = np.random.random()
                sigma = np.random.random()
                if np.square(m) < q + delta * q:
                    break

            initial = [m, q, sigma]

            _, _, _ = state_equations_convergence(
                fpe.var_func_L2,
                fpe.var_hat_func_L1_num,
                delta=delta,
                lamb=l,
                alpha=alpha,
                init=initial,
                verbose=True,
            )
