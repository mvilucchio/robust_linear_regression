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
        / ((V + delta_small) * (V + delta_large))
        / (
            eps
            * np.exp(((y - omega) ** 2) / (2 * (V + delta_small)))
            * np.sqrt(V + delta_small)
            + (1 - eps)
            * np.exp(((y - omega) ** 2) / (2 * (V + delta_large)))
            * np.sqrt(V + delta_large)
        )
        * (
            eps
            * np.exp(((y - omega) ** 2) / (2 * (V + delta_small)))
            * np.sqrt((V + delta_small) ** 3)
            + (1 - eps)
            * np.exp(((y - omega) ** 2) / (2 * (V + delta_large)))
            * np.sqrt((V + delta_large) ** 3)
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
    eta = m ** 2 / q
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
    alpha = 7.4
    deltas = [[1.0, 10.0]]
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
                fpedb.var_hat_func_L2_num_eps,
                delta_small=delta_small,
                delta_large=delta_large,
                lambd=l,
                alpha=alpha,
                init=initial,
                verbose=True,
            )
