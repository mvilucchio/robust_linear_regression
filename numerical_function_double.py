from tqdm.auto import tqdm

# from cv2 import integral
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from tqdm.auto import tqdm
from scipy.integrate import dblquad, quad
import fixed_point_equations as fpe
import numba as nb
import numerical_functions as numfun
import fixed_point_equations_double as fpedb

MULT_INTEGRAL = 5
A_HUBER = 1.0
EPS = 0.1


@nb.njit(error_model="numpy", fastmath=True)
def ZoutBayes_eps(y, omega, V, delta_small, delta_large, eps):
    return (1 - eps) * np.exp(-((y - omega) ** 2) / (2 * (V + delta_small))) / np.sqrt(
        2 * np.pi * (V + delta_small)
    ) + eps * np.exp(-((y - omega) ** 2) / (2 * (V + delta_large))) / np.sqrt(
        2 * np.pi * (V + delta_large)
    )


@nb.njit(error_model="numpy", fastmath=True)
def foutBayes_eps(y, omega, V, delta_small, delta_large, eps):
    return (
        (y - omega)
        * (
            (1 - eps)
            * np.exp(-((y - omega) ** 2) / (2 * (V + delta_small)))
            / np.power(V + delta_small, 3 / 2)
            + eps
            * np.exp(-((y - omega) ** 2) / (2 * (V + delta_large)))
            / np.power(V + delta_large, 3 / 2)
        )
        / (
            (1 - eps)
            * np.exp(-((y - omega) ** 2) / (2 * (V + delta_small)))
            / np.power(V + delta_small, 1 / 2)
            + eps
            * np.exp(-((y - omega) ** 2) / (2 * (V + delta_large)))
            / np.power(V + delta_large, 1 / 2)
        )
    )


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


# -----


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


# -----


@nb.njit(error_model="numpy", fastmath=True)
def q_integral_BO_eps(y, xi, q, m, sigma, delta_small, delta_large, eps):
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_eps(y, np.sqrt(q) * xi, 1 - q, delta_small, delta_large, eps)
        * (foutBayes_eps(y, np.sqrt(q) * xi, 1 - q, delta_small, delta_large, eps) ** 2)
    )


# -----


@nb.njit(error_model="numpy", fastmath=True)
def m_integral_L2_eps(y, xi, q, m, sigma, delta_small, delta_large, eps):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_eps(y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps)
        * foutBayes_eps(y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps)
        * foutL2(y, np.sqrt(q) * xi, sigma)
    )


@nb.njit(error_model="numpy", fastmath=True)
def q_integral_L2_eps(y, xi, q, m, sigma, delta_small, delta_large, eps):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_eps(y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps)
        * (foutL2(y, np.sqrt(q) * xi, sigma) ** 2)
    )


@nb.njit(error_model="numpy", fastmath=True)
def sigma_integral_L2_eps(y, xi, q, m, sigma, delta_small, delta_large, eps):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_eps(y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps)
        * DfoutL2(y, np.sqrt(q) * xi, sigma)
    )


# -----


@nb.njit(error_model="numpy", fastmath=True)
def m_integral_Huber_eps(
    y, xi, q, m, sigma, delta_small, delta_large, eps=EPS, a=A_HUBER
):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_eps(y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps)
        * foutBayes_eps(y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps)
        * foutHuber(y, np.sqrt(q) * xi, sigma, a=a)
    )


@nb.njit(error_model="numpy", fastmath=True)
def q_integral_Huber_eps(
    y, xi, q, m, sigma, delta_small, delta_large, eps=EPS, a=A_HUBER
):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_eps(y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps)
        * (foutHuber(y, np.sqrt(q) * xi, sigma, a=a) ** 2)
    )


@nb.njit(error_model="numpy", fastmath=True)
def sigma_integral_Huber_eps(
    y, xi, q, m, sigma, delta_small, delta_large, eps=EPS, a=A_HUBER
):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_eps(y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps)
        * DfoutHuber(y, np.sqrt(q) * xi, sigma, a=a)
    )


# -----


def q_hat_equation_BO_eps(m, q, sigma, delta_small, delta_large, eps=EPS):
    borders = find_integration_borders(
        lambda y, xi: q_integral_BO_eps(
            y, xi, q, m, sigma, delta_small, delta_large, eps
        ),
        np.sqrt((1 + delta_small)),
        1.0,
    )
    return dblquad(
        q_integral_BO_eps,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        borders[1][1],
        args=(q, m, sigma, delta_small, delta_large, eps),
    )[0]


# -----


def m_hat_equation_L2_eps(m, q, sigma, delta_small, delta_large, eps=EPS):
    borders = find_integration_borders(
        lambda y, xi: m_integral_L2_eps(
            y, xi, q, m, sigma, delta_small, delta_large, eps
        ),
        np.sqrt((1 + delta_small)),
        1.0,
    )
    return dblquad(
        m_integral_L2_eps,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        borders[1][1],
        args=(q, m, sigma, delta_small, delta_large, eps),
    )[0]


def q_hat_equation_L2_eps(m, q, sigma, delta_small, delta_large, eps=EPS):
    borders = find_integration_borders(
        lambda y, xi: q_integral_L2_eps(
            y, xi, q, m, sigma, delta_small, delta_large, eps
        ),
        np.sqrt((1 + delta_small)),
        1.0,
    )
    return dblquad(
        q_integral_L2_eps,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        borders[1][1],
        args=(q, m, sigma, delta_small, delta_large, eps),
    )[0]


def sigma_hat_equation_L2_eps(m, q, sigma, delta_small, delta_large, eps=EPS):
    borders = find_integration_borders(
        lambda y, xi: sigma_integral_L2_eps(
            y, xi, q, m, sigma, delta_small, delta_large, eps
        ),
        np.sqrt((1 + delta_small)),
        1.0,
    )
    return dblquad(
        sigma_integral_L2_eps,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        borders[1][1],
        args=(q, m, sigma, delta_small, delta_large, eps),
    )[0]


# -----------


def border_plus_Huber(xi, m, q, sigma, a=A_HUBER):
    return np.sqrt(q) / a * xi + (sigma + 1)


def border_minus_Huber(xi, m, q, sigma, a=A_HUBER):
    return np.sqrt(q) / a * xi - (sigma + 1)


def test_fun_upper_Huber(y, m, q, sigma, a=A_HUBER):
    return a / np.sqrt(q) * (-(sigma + 1) + y)


def test_fun_down_Huber(y, m, q, sigma, a=A_HUBER):
    return a / np.sqrt(q) * ((sigma + 1) + y)


def integral_fpe(
    integral_form,
    border_fun_plus,
    border_fun_minus,
    test_function,
    m,
    q,
    sigma,
    delta_small,
    delta_large,
    eps=EPS,
):
    borders = find_integration_borders(
        lambda y, xi: integral_form(
            y, xi, q, m, sigma, delta_small, delta_large, eps=EPS
        ),
        np.sqrt(1 + delta_small),
        1.0,
    )

    [_, max_val], _ = borders

    # print("border is: ", max_val)

    xi_test = test_function(max_val, m, q, sigma)
    xi_test_2 = test_function(-max_val, m, q, sigma)
    # print("xi_test : {} xi_test_2 : {} max_val : {}".format(
    #     xi_test, xi_test_2, max_val))

    if xi_test > max_val:
        # print("case 1")
        domain_xi = [[-max_val, max_val]] * 3
        domain_y = [
            [lambda xi: border_fun_plus(xi, m, q, sigma), lambda xi: max_val],
            [
                lambda xi: border_fun_minus(xi, m, q, sigma),
                lambda xi: border_fun_plus(xi, m, q, sigma),
            ],
            [lambda xi: -max_val, lambda xi: border_fun_minus(xi, m, q, sigma)],
        ]
    elif xi_test >= 0:
        xi_test_2 = test_function(-max_val, m, q, sigma)
        if xi_test_2 < -max_val:
            # print("case 2.A")
            domain_xi = [
                [-max_val, xi_test],
                [-xi_test, max_val],
                [-max_val, -xi_test],
                [xi_test, max_val],
                [-xi_test, xi_test],
            ]
            domain_y = [
                [lambda xi: border_fun_plus(xi, m, q, sigma), lambda xi: max_val],
                [lambda xi: -max_val, lambda xi: border_fun_minus(xi, m, q, sigma),],
                [lambda xi: -max_val, lambda xi: border_fun_plus(xi, m, q, sigma)],
                [lambda xi: border_fun_minus(xi, m, q, sigma), lambda xi: max_val],
                [
                    lambda xi: border_fun_minus(xi, m, q, sigma),
                    lambda xi: border_fun_plus(xi, m, q, sigma),
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
                [lambda xi: border_fun_plus(xi, m, q, sigma), lambda xi: max_val],
                [lambda xi: -max_val, lambda xi: border_fun_minus(xi, m, q, sigma),],
                [lambda xi: -max_val, lambda xi: border_fun_plus(xi, m, q, sigma)],
                [lambda xi: border_fun_minus(xi, m, q, sigma), lambda xi: max_val],
                [
                    lambda xi: border_fun_minus(xi, m, q, sigma),
                    lambda xi: border_fun_plus(xi, m, q, sigma),
                ],
                [lambda xi: -max_val, lambda xi: max_val],
                [lambda xi: -max_val, lambda xi: max_val],
            ]
    elif xi_test > -max_val:
        xi_test_2 = test_function(-max_val, m, q, sigma)
        if xi_test_2 < -max_val:
            # print("case 3.A")
            domain_xi = [
                [-max_val, xi_test],
                [-xi_test, max_val],
                [-max_val, xi_test],
                [-xi_test, max_val],
                [xi_test, -xi_test],
            ]
            domain_y = [
                [lambda xi: border_fun_plus(xi, m, q, sigma), lambda xi: max_val],
                [lambda xi: -max_val, lambda xi: border_fun_minus(xi, m, q, sigma),],
                [lambda xi: -max_val, lambda xi: border_fun_plus(xi, m, q, sigma)],
                [lambda xi: border_fun_minus(xi, m, q, sigma), lambda xi: max_val],
                [lambda xi: -max_val, lambda xi: max_val],
            ]
        else:
            # print("case 3.B")
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
                [lambda xi: border_fun_plus(xi, m, q, sigma), lambda xi: max_val],
                [lambda xi: -max_val, lambda xi: border_fun_minus(xi, m, q, sigma),],
                [lambda xi: -max_val, lambda xi: border_fun_plus(xi, m, q, sigma)],
                [lambda xi: border_fun_minus(xi, m, q, sigma), lambda xi: max_val],
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
            args=(q, m, sigma, delta_small, delta_large, eps),
        )[0]
    return integral_value


# ------------------


def m_hat_equation_Huber_eps(m, q, sigma, delta_small, delta_large, eps=EPS, a=A_HUBER):
    borders = find_integration_borders(
        lambda y, xi: m_integral_Huber_eps(
            y, xi, q, m, sigma, delta_small, delta_large, eps, a
        ),
        np.sqrt((1 + delta_small)),
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
            m_integral_Huber_eps,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, a),
        )[0]

    return integral_value


def q_hat_equation_Huber_eps(m, q, sigma, delta_small, delta_large, eps=EPS, a=A_HUBER):
    borders = find_integration_borders(
        lambda y, xi: q_integral_Huber_eps(
            y, xi, q, m, sigma, delta_small, delta_large, eps, a
        ),
        np.sqrt((1 + delta_small)),
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
            q_integral_Huber_eps,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, a),
        )[0]
    return integral_value


def sigma_hat_equation_Huber_eps(
    m, q, sigma, delta_small, delta_large, eps=EPS, a=A_HUBER
):
    borders = find_integration_borders(
        lambda y, xi: sigma_integral_Huber_eps(
            y, xi, q, m, sigma, delta_small, delta_large, eps, a
        ),
        np.sqrt((1 + delta_small)),
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
            sigma_integral_Huber_eps,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, a),
        )[0]
    return integral_value


# -------------------


def state_equations_convergence(
    var_func,
    var_hat_func,
    delta_small=0.1,
    delta_large=1.0,
    lambd=0.01,
    alpha=0.5,
    eps=0.1,
    init=(0.5, 0.5, 0.5),
    verbose=False,
):
    m, q, sigma = init[0], init[1], init[2]
    err = 1.0
    blend = 0.5
    iter = 0
    while err > 1e-6:
        m_hat, q_hat, sigma_hat = var_hat_func(
            m, q, sigma, alpha, delta_small, delta_large, eps=eps
        )

        temp_m, temp_q, temp_sigma = m, q, sigma

        m, q, sigma = var_func(
            m_hat, q_hat, sigma_hat, alpha, delta_small, delta_large, lambd
        )

        err = np.max(np.abs([(temp_m - m), (temp_q - q), (temp_sigma - sigma)]))

        m = blend * m + (1 - blend) * temp_m
        q = blend * q + (1 - blend) * temp_q
        sigma = blend * sigma + (1 - blend) * temp_sigma
        if verbose:
            print("i : {} m : {} q : {} sigma : {}".format(iter, m, q, sigma))

        iter += 1
    return m, q, sigma


if __name__ == "__main__":
    # test the convergence
    alpha = 0.05
    deltas = [[0.5, 1.5]]
    lambdas = [1.0]

    for idx, l in enumerate(tqdm(lambdas, desc="lambda", leave=False)):
        for jdx, [delta_small, delta_large] in enumerate(
            tqdm(deltas, desc="delta", leave=False)
        ):
            i = idx * len(deltas) + jdx

            while True:
                m = np.random.random()
                q = np.random.random()
                sigma = np.random.random()
                if (
                    np.square(m) < q + delta_small * q
                    and np.square(m) < q + delta_large * q
                ):
                    break

            initial = [m, q, sigma]

            _, _, _ = state_equations_convergence(
                fpedb.var_func_L2,
                fpedb.var_hat_func_Huber_num_eps,
                delta_small=delta_small,
                delta_large=delta_large,
                lambd=l,
                alpha=alpha,
                init=initial,
                verbose=True,
            )
