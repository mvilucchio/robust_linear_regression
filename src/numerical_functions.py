import numpy as np
from scipy.integrate import dblquad
from numba import njit, vectorize
from src.integration_utils import (
    find_integration_borders_square,
    divide_integration_borders_grid,
    domains_double_line_constraint,
    domains_double_line_constraint_only_inside,
)

MULT_INTEGRAL = 10
EPSABS = 1e-9
EPSREL = 1e-9


@njit(error_model="numpy", fastmath=True)
def ZoutBayes_single_noise(y, omega, V, delta):
    return np.exp(-((y - omega) ** 2) / (2 * (V + delta))) / np.sqrt(
        2 * np.pi * (V + delta)
    )


@njit(error_model="numpy", fastmath=True)
def foutBayes_single_noise(y, omega, V, delta):
    return (y - omega) / (V + delta)


@njit(error_model="numpy", fastmath=True)
def ZoutBayes_double_noise(y, omega, V, delta_small, delta_large, eps):
    return (1 - eps) * np.exp(-((y - omega) ** 2) / (2 * (V + delta_small))) / np.sqrt(
        2 * np.pi * (V + delta_small)
    ) + eps * np.exp(-((y - omega) ** 2) / (2 * (V + delta_large))) / np.sqrt(
        2 * np.pi * (V + delta_large)
    )


@njit(error_model="numpy", fastmath=True)
def foutBayes_double_noise(y, omega, V, delta_small, delta_large, eps):
    small_exponential = np.exp(-((y - omega) ** 2) / (2 * (V + delta_small)))
    large_exponential = np.exp(-((y - omega) ** 2) / (2 * (V + delta_large)))
    return (
        (y - omega)
        * (
            (1 - eps) * small_exponential / np.power(V + delta_small, 3 / 2)
            + eps * large_exponential / np.power(V + delta_large, 3 / 2)
        )
        / (
            (1 - eps) * small_exponential / np.power(V + delta_small, 1 / 2)
            + eps * large_exponential / np.power(V + delta_large, 1 / 2)
        )
    )

@njit(error_model="numpy", fastmath=True)
def ZoutBayes_decorrelated_noise(y, omega, V, delta_small, delta_large, eps, beta):
    return (1 - eps) * np.exp( -((y - omega) ** 2) / (2 * (V + delta_small)) ) / np.sqrt(
        2 * np.pi * (V + delta_small)
    ) + eps * np.exp(-((y - beta * omega) ** 2) / (2 * (beta ** 2 * V + delta_large))) / np.sqrt(
        2 * np.pi * (beta ** 2 * V + delta_large)
    )

@njit(error_model="numpy", fastmath=True)
def foutBayes_decorrelated_noise(y, omega, V, delta_small, delta_large, eps, beta):
    small_exponential = np.exp(-((y - omega) ** 2) / (2 * (V + delta_small)))
    large_exponential = np.exp(-((y - beta * omega) ** 2) / (2 * (beta**2 * V + delta_large)))
    return (
        (
            (y - omega) * (1 - eps) * small_exponential / np.power(V + delta_small, 3 / 2)
            + eps * beta * (y - beta * omega) * large_exponential / np.power(beta ** 2 * V + delta_large, 3 / 2)
        )
        / (
            (1 - eps) * small_exponential / np.power(V + delta_small, 1 / 2)
            + eps * large_exponential / np.power(beta ** 2 * V + delta_large, 1 / 2)
        )
    )


# -------


@njit(error_model="numpy", fastmath=True)
def foutL2(y, omega, V):
    return (y - omega) / (1 + V)


@njit(error_model="numpy", fastmath=True)
def DfoutL2(y, omega, V):
    return -1.0 / (1 + V)


@njit(error_model="numpy", fastmath=True)
def foutL1(y, omega, V):
    return (y - omega + np.sign(omega - y) * np.maximum(np.abs(omega - y) - V, 0.0)) / V


@njit(error_model="numpy", fastmath=True)
def DfoutL1(y, omega, V):
    if np.abs(omega - y) > V:
        return 0.0
    else:
        return -1.0 / V


@vectorize
def foutHuber(y, omega, V, a):
    if a + a * V + omega < y:
        return a
    elif np.abs(y - omega) <= a + a * V:
        return (y - omega) / (1 + V)
    elif omega > a + a * V + y:
        return -a
    else:
        return 0.0


@vectorize
def DfoutHuber(y, omega, V, a):
    if (y < omega and a + a * V + y < omega) or (a + a * V + omega < y):
        return 0.0
    else:
        return -1.0 / (1 + V)


# --------------
# Functions to integrate - Single Noise
# --------------


@njit(error_model="numpy", fastmath=True)
def q_integral_BO_single_noise(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_single_noise(y, np.sqrt(q) * xi, 1 - q, delta)
        * (foutBayes_single_noise(y, np.sqrt(q) * xi, 1 - q, delta) ** 2)
    )


# ----


@njit(error_model="numpy", fastmath=True)
def m_integral_L2_single_noise(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * foutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * foutL2(y, np.sqrt(q) * xi, sigma)
    )


@njit(error_model="numpy", fastmath=True)
def q_integral_L2_single_noise(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * (foutL2(y, np.sqrt(q) * xi, sigma) ** 2)
    )


@njit(error_model="numpy", fastmath=True)
def sigma_integral_L2_single_noise(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * DfoutL2(y, np.sqrt(q) * xi, sigma)
    )


# ----


@njit(error_model="numpy", fastmath=True)
def m_integral_L1_single_noise(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * foutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * foutL1(y, np.sqrt(q) * xi, sigma)
    )


@njit(error_model="numpy", fastmath=True)
def q_integral_L1_single_noise(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * (foutL1(y, np.sqrt(q) * xi, sigma) ** 2)
    )


@njit(error_model="numpy", fastmath=True)
def sigma_integral_L1_single_noise(y, xi, q, m, sigma, delta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * DfoutL1(y, np.sqrt(q) * xi, sigma)
    )


# ----


@njit(error_model="numpy", fastmath=True)
def m_integral_Huber_single_noise(y, xi, q, m, sigma, delta, a):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * foutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * foutHuber(y, np.sqrt(q) * xi, sigma, a)
    )


@njit(error_model="numpy", fastmath=True)
def q_integral_Huber_single_noise(y, xi, q, m, sigma, delta, a):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * (foutHuber(y, np.sqrt(q) * xi, sigma, a) ** 2)
    )


@njit(error_model="numpy", fastmath=True)
def sigma_integral_Huber_single_noise(y, xi, q, m, sigma, delta, a):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_single_noise(y, np.sqrt(eta) * xi, (1 - eta), delta)
        * DfoutHuber(y, np.sqrt(q) * xi, sigma, a)
    )


# --------------
# Functions to integrate - Double Noise
# --------------


@njit(error_model="numpy", fastmath=True)
def q_integral_BO_double_noise(y, xi, q, m, sigma, delta_small, delta_large, eps):
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_double_noise(
            y, np.sqrt(q) * xi, (1 - q), delta_small, delta_large, eps
        )
        * (
            foutBayes_double_noise(
                y, np.sqrt(q) * xi, (1 - q), delta_small, delta_large, eps
            )
            ** 2
        )
    )


#  ----


@njit(error_model="numpy", fastmath=True)
def m_integral_L2_double_noise(y, xi, q, m, sigma, delta_small, delta_large, eps):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * foutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * foutL2(y, np.sqrt(q) * xi, sigma)
    )


@njit(error_model="numpy", fastmath=True)
def q_integral_L2_double_noise(y, xi, q, m, sigma, delta_small, delta_large, eps):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * (foutL2(y, np.sqrt(q) * xi, sigma) ** 2)
    )


@njit(error_model="numpy", fastmath=True)
def sigma_integral_L2_double_noise(y, xi, q, m, sigma, delta_small, delta_large, eps):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * DfoutL2(y, np.sqrt(q) * xi, sigma)
    )


# ----


@njit(error_model="numpy", fastmath=True)
def m_integral_Huber_double_noise(y, xi, q, m, sigma, delta_small, delta_large, eps, a):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * foutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * foutHuber(y, np.sqrt(q) * xi, sigma, a)
    )


@njit(error_model="numpy", fastmath=True)
def q_integral_Huber_double_noise(y, xi, q, m, sigma, delta_small, delta_large, eps, a):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * (foutHuber(y, np.sqrt(q) * xi, sigma, a) ** 2)
    )


@njit(error_model="numpy", fastmath=True)
def sigma_integral_Huber_double_noise(
    y, xi, q, m, sigma, delta_small, delta_large, eps, a
):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_double_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps
        )
        * DfoutHuber(y, np.sqrt(q) * xi, sigma, a)
    )

# --------------
# Functions to integrate - Decorrelated Noise
# --------------

@njit(error_model="numpy", fastmath=True)
def q_integral_BO_decorrelated_noise(y, xi, q, m, sigma, delta_small, delta_large, eps, beta):
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_decorrelated_noise(
            y, np.sqrt(q) * xi, (1 - q), delta_small, delta_large, eps, beta
        )
        * (
            foutBayes_decorrelated_noise(
                y, np.sqrt(q) * xi, (1 - q), delta_small, delta_large, eps, beta
            )
            ** 2
        )
    )


#  ----


@njit(error_model="numpy", fastmath=True)
def m_integral_L2_decorrelated_noise(y, xi, q, m, sigma, delta_small, delta_large, eps, beta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_decorrelated_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps, beta
        )
        * foutBayes_decorrelated_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps, beta
        )
        * foutL2(y, np.sqrt(q) * xi, sigma)
    )


@njit(error_model="numpy", fastmath=True)
def q_integral_L2_decorrelated_noise(y, xi, q, m, sigma, delta_small, delta_large, eps, beta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_decorrelated_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps, beta
        )
        * (foutL2(y, np.sqrt(q) * xi, sigma) ** 2)
    )


@njit(error_model="numpy", fastmath=True)
def sigma_integral_L2_decorrelated_noise(y, xi, q, m, sigma, delta_small, delta_large, eps, beta):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_decorrelated_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps, beta
        )
        * DfoutL2(y, np.sqrt(q) * xi, sigma)
    )


# ----


@njit(error_model="numpy", fastmath=True)
def m_integral_Huber_decorrelated_noise(y, xi, q, m, sigma, delta_small, delta_large, eps, beta, a):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_decorrelated_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps, beta
        )
        * foutBayes_decorrelated_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps, beta
        )
        * foutHuber(y, np.sqrt(q) * xi, sigma, a)
    )


@njit(error_model="numpy", fastmath=True)
def q_integral_Huber_decorrelated_noise(y, xi, q, m, sigma, delta_small, delta_large, eps, beta, a):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_decorrelated_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps, beta
        )
        * (foutHuber(y, np.sqrt(q) * xi, sigma, a) ** 2)
    )


@njit(error_model="numpy", fastmath=True)
def sigma_integral_Huber_decorrelated_noise(
    y, xi, q, m, sigma, delta_small, delta_large, eps, beta, a
):
    eta = m ** 2 / q
    return (
        np.exp(-(xi ** 2) / 2)
        / np.sqrt(2 * np.pi)
        * ZoutBayes_decorrelated_noise(
            y, np.sqrt(eta) * xi, (1 - eta), delta_small, delta_large, eps, beta
        )
        * DfoutHuber(y, np.sqrt(q) * xi, sigma, a)
    )


# -----------


def border_plus_L1(xi, m, q, sigma):
    return np.sqrt(q) * xi + sigma


def border_minus_L1(xi, m, q, sigma):
    return np.sqrt(q) * xi - sigma


def test_fun_upper_L1(y, m, q, sigma):
    return (y - sigma) / np.sqrt(q)


def test_fun_down_L1(y, m, q, sigma):
    return (y + sigma) / np.sqrt(q)


def border_plus_Huber(xi, m, q, sigma, a):
    return np.sqrt(q) * xi + a * (sigma + 1)


def border_minus_Huber(xi, m, q, sigma, a):
    return np.sqrt(q) * xi - a * (sigma + 1)


def test_fun_upper_Huber(y, m, q, sigma, a):
    return 1 / np.sqrt(q) * (-a * (sigma + 1) + y)


def test_fun_down_Huber(y, m, q, sigma, a):
    return 1 / np.sqrt(q) * (a * (sigma + 1) + y)


# ------------------
# BayesOpt equations single noise
# ------------------


def q_hat_equation_BO_single_noise(m, q, sigma, delta):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_BO_single_noise(y, xi, q, m, sigma, delta),
        np.sqrt((1 + delta)),
        1.0,
    )
    return dblquad(
        q_integral_BO_single_noise,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        borders[1][1],
        args=(q, m, sigma, delta),
        epsabs=EPSABS,
        epsrel=EPSREL
    )[0]


# ------------------
# L2 equations single noise
# ------------------


def m_hat_equation_L2_single_noise(m, q, sigma, delta):
    borders = find_integration_borders_square(
        lambda y, xi: m_integral_L2_single_noise(y, xi, q, m, sigma, delta),
        np.sqrt((1 + delta)),
        1.0,
    )
    return dblquad(
        m_integral_L2_single_noise,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        borders[1][1],
        args=(q, m, sigma, delta),
        epsabs=EPSABS,
        epsrel=EPSREL
    )[0]


def q_hat_equation_L2_single_noise(m, q, sigma, delta):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_L2_single_noise(y, xi, q, m, sigma, delta),
        np.sqrt((1 + delta)),
        1.0,
    )
    return dblquad(
        q_integral_L2_single_noise,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        borders[1][1],
        args=(q, m, sigma, delta),
        epsabs=EPSABS,
        epsrel=EPSREL
    )[0]


def sigma_hat_equation_L2_single_noise(m, q, sigma, delta):
    borders = find_integration_borders_square(
        lambda y, xi: sigma_integral_L2_single_noise(y, xi, q, m, sigma, delta),
        np.sqrt((1 + delta)),
        1.0,
    )
    return dblquad(
        sigma_integral_L2_single_noise,
        borders[0][0],
        borders[0][1],
        borders[1][0],
        borders[1][1],
        args=(q, m, sigma, delta),
        epsabs=EPSABS,
        epsrel=EPSREL
    )[0]


# ------------------
# L1 equations single noise
# ------------------

def m_hat_equation_L1_single_noise(m, q, sigma, delta):
    borders = find_integration_borders_square(
        lambda y, xi: m_integral_L1_single_noise(y, xi, q, m, sigma, delta),
        np.sqrt((1 + delta)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma}
    domain_xi, domain_y = domains_double_line_constraint(
        borders,
        border_plus_L1,
        border_minus_L1,
        test_fun_upper_L1,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            m_integral_L1_single_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta),
            epsabs=EPSABS,
            epsrel=EPSREL
        )[0]

    return integral_value


def q_hat_equation_L1_single_noise(m, q, sigma, delta):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_L1_single_noise(y, xi, q, m, sigma, delta),
        np.sqrt((1 + delta)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma}
    domain_xi, domain_y = domains_double_line_constraint(
        borders,
        border_plus_L1,
        border_minus_L1,
        test_fun_upper_L1,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_L1_single_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta),
            epsabs=EPSABS,
            epsrel=EPSREL
        )[0]

    return integral_value


def sigma_hat_equation_L1_single_noise(m, q, sigma, delta):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_L1_single_noise(y, xi, q, m, sigma, delta),
        np.sqrt((1 + delta)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma}
    domain_xi, domain_y = domains_double_line_constraint(
        borders,
        border_plus_L1,
        border_minus_L1,
        test_fun_upper_L1,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            sigma_integral_L1_single_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta),
            epsabs=EPSABS,
            epsrel=EPSREL
        )[0]

    return integral_value


# ------------------
# Huber equations single noise
# ------------------


def m_hat_equation_Huber_single_noise(m, q, sigma, delta, a):
    borders = find_integration_borders_square(
        lambda y, xi: m_integral_Huber_single_noise(y, xi, q, m, sigma, delta, a),
        np.sqrt((1 + delta)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma, "a": a}
    domain_xi, domain_y = domains_double_line_constraint(
        borders,
        border_plus_Huber,
        border_minus_Huber,
        test_fun_upper_Huber,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            m_integral_Huber_single_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta, a),
            epsabs=EPSABS,
            epsrel=EPSREL
        )[0]

    return integral_value


def q_hat_equation_Huber_single_noise(m, q, sigma, delta, a):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_Huber_single_noise(y, xi, q, m, sigma, delta, a),
        np.sqrt((1 + delta)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma, "a": a}
    domain_xi, domain_y = domains_double_line_constraint(
        borders,
        border_plus_Huber,
        border_minus_Huber,
        test_fun_upper_Huber,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_Huber_single_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta, a),
            epsabs=EPSABS,
            epsrel=EPSREL
        )[0]

    return integral_value


def sigma_hat_equation_Huber_single_noise(m, q, sigma, delta, a):
    borders = find_integration_borders_square(
        lambda y, xi: sigma_integral_Huber_single_noise(y, xi, q, m, sigma, delta, a),
        np.sqrt((1 + delta)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma, "a": a}
    domain_xi, domain_y = domains_double_line_constraint_only_inside(
        borders,
        border_plus_Huber,
        border_minus_Huber,
        test_fun_upper_Huber,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            sigma_integral_Huber_single_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta, a),
            epsabs=EPSABS,
            epsrel=EPSREL
        )[0]

    return integral_value


# ------------------
# BayesOpt equations double noise
# ------------------


def q_hat_equation_BO_double_noise(m, q, sigma, delta_small, delta_large, eps):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_BO_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps
        ),
        np.sqrt((1 + delta_small)),
        1.0,
    )

    domain_xi, domain_y = divide_integration_borders_grid(borders)

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_BO_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps),
            epsabs=EPSABS,
            epsrel=EPSREL
        )[0]

    return integral_value


# ------------------
# L2 equations double noise
# ------------------


def m_hat_equation_L2_double_noise(m, q, sigma, delta_small, delta_large, eps):
    borders = find_integration_borders_square(
        lambda y, xi: m_integral_L2_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps
        ),
        np.sqrt((1 + delta_small)),
        1.0,
    )

    domain_xi, domain_y = divide_integration_borders_grid(borders)

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            m_integral_L2_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps),
            epsabs=EPSABS,
            epsrel=EPSREL
        )[0]

    return integral_value


def q_hat_equation_L2_double_noise(m, q, sigma, delta_small, delta_large, eps):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_L2_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps
        ),
        np.sqrt((1 + delta_small)),
        1.0,
    )

    domain_xi, domain_y = divide_integration_borders_grid(borders)

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_L2_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps),
            epsabs=EPSABS,
            epsrel=EPSREL
        )[0]

    return integral_value


def sigma_hat_equation_L2_double_noise(m, q, sigma, delta_small, delta_large, eps):
    borders = find_integration_borders_square(
        lambda y, xi: sigma_integral_L2_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps
        ),
        np.sqrt((1 + delta_small)),
        1.0,
    )

    domain_xi, domain_y = divide_integration_borders_grid(borders)

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            sigma_integral_L2_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps),
            epsabs=EPSABS,
            epsrel=EPSREL
        )[0]

    return integral_value


# ------------------
# L1 equations double noise
# ------------------


# ------------------
#  Huber equations double noise
# ------------------


def m_hat_equation_Huber_double_noise(m, q, sigma, delta_small, delta_large, eps, a):
    borders = find_integration_borders_square(
        lambda y, xi: m_integral_Huber_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, a
        ),
        np.sqrt((1 + delta_small)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma, "a": a}
    domain_xi, domain_y = domains_double_line_constraint(
        borders,
        border_plus_Huber,
        border_minus_Huber,
        test_fun_upper_Huber,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            m_integral_Huber_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, a),
            epsabs=EPSABS,
            epsrel=EPSREL
        )[0]

    return integral_value


def q_hat_equation_Huber_double_noise(m, q, sigma, delta_small, delta_large, eps, a):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_Huber_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, a
        ),
        np.sqrt((1 + delta_small)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma, "a": a}
    domain_xi, domain_y = domains_double_line_constraint(
        borders,
        border_plus_Huber,
        border_minus_Huber,
        test_fun_upper_Huber,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_Huber_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, a),
            epsabs=EPSABS,
            epsrel=EPSREL
        )[0]

    return integral_value


def sigma_hat_equation_Huber_double_noise(m, q, sigma, delta_small, delta_large, eps, a):
    borders = find_integration_borders_square(
        lambda y, xi: sigma_integral_Huber_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, a
        ),
        np.sqrt((1 + delta_small)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma, "a": a}
    domain_xi, domain_y = domains_double_line_constraint_only_inside(
        borders,
        border_plus_Huber,
        border_minus_Huber,
        test_fun_upper_Huber,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            sigma_integral_Huber_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, a),
            epsabs=EPSABS,
            epsrel=EPSREL
        )[0]

    return integral_value

# ------------------
# BayesOpt equations decorrelated noise
# ------------------

def q_hat_equation_BO_decorrelated_noise(m, q, sigma, delta_small, delta_large, eps, beta):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_BO_decorrelated_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, beta
        ),
        np.sqrt((1 + delta_small)),
        1.0,
    )

    domain_xi, domain_y = divide_integration_borders_grid(borders)

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_BO_decorrelated_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, beta),
        )[0]

    return integral_value

# ------------------
# L2 equations decorrelated noise
# ------------------

def m_hat_equation_L2_decorrelated_noise(m, q, sigma, delta_small, delta_large, eps, beta):
    borders = find_integration_borders_square(
        lambda y, xi: m_integral_L2_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, beta
        ),
        np.sqrt((1 + delta_small)),
        1.0,
    )

    domain_xi, domain_y = divide_integration_borders_grid(borders)

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            m_integral_L2_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, beta),
        )[0]

    return integral_value


def q_hat_equation_L2_decorrelated_noise(m, q, sigma, delta_small, delta_large, eps, beta):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_L2_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, beta
        ),
        np.sqrt((1 + delta_small)),
        1.0,
    )

    domain_xi, domain_y = divide_integration_borders_grid(borders)

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_L2_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, beta),
        )[0]

    return integral_value


def sigma_hat_equation_L2_decorrelated_noise(m, q, sigma, delta_small, delta_large, eps, beta):
    borders = find_integration_borders_square(
        lambda y, xi: sigma_integral_L2_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, beta
        ),
        np.sqrt((1 + delta_small)),
        1.0,
    )

    domain_xi, domain_y = divide_integration_borders_grid(borders)

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            sigma_integral_L2_double_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, beta),
        )[0]

    return integral_value

# ------------------
# L1 equations decorrelated noise
# ------------------

# ------------------
#  Huber equations decorrelated noise
# ------------------

def m_hat_equation_Huber_decorrelated_noise(m, q, sigma, delta_small, delta_large, eps, beta, a):
    borders = find_integration_borders_square(
        lambda y, xi: m_integral_Huber_decorrelated_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, beta, a
        ),
        np.sqrt((1 + delta_small)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma, "a": a}
    domain_xi, domain_y = domains_double_line_constraint(
        borders,
        border_plus_Huber,
        border_minus_Huber,
        test_fun_upper_Huber,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            m_integral_Huber_decorrelated_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, beta, a),
        )[0]

    return integral_value


def q_hat_equation_Huber_decorrelated_noise(m, q, sigma, delta_small, delta_large, eps, beta, a):
    borders = find_integration_borders_square(
        lambda y, xi: q_integral_Huber_decorrelated_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, beta, a
        ),
        np.sqrt((1 + delta_small)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma, "a": a}
    domain_xi, domain_y = domains_double_line_constraint(
        borders,
        border_plus_Huber,
        border_minus_Huber,
        test_fun_upper_Huber,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_Huber_decorrelated_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, beta, a),
        )[0]

    return integral_value


def sigma_hat_equation_Huber_decorrelated_noise(m, q, sigma, delta_small, delta_large, eps, beta, a):
    borders = find_integration_borders_square(
        lambda y, xi: sigma_integral_Huber_decorrelated_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps, beta, a
        ),
        np.sqrt((1 + delta_small)),
        1.0,
    )

    args = {"m": m, "q": q, "sigma": sigma, "a": a}
    domain_xi, domain_y = domains_double_line_constraint(
        borders,
        border_plus_Huber,
        border_minus_Huber,
        test_fun_upper_Huber,
        args,
        args,
        args,
    )

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            sigma_integral_Huber_decorrelated_noise,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps, beta, a),
        )[0]

    return integral_value