import numpy as np
from tqdm.auto import tqdm
import src.numerical_functions as numfun

BLEND = 0.5
TOL_FPE = 1e-6


def state_equations(
    var_func,
    var_hat_func,
    reg_param=0.01,
    alpha=0.5,
    init=(0.5, 0.5, 0.5),
    noise_kwargs={"delta_small": 0.1, "delta_large": 2.0, "percentage": 0.1},
):
    m, q, sigma = init[0], init[1], init[2]
    err = 1.0
    blend = BLEND
    while err > TOL_FPE:
        m_hat, q_hat, sigma_hat = var_hat_func(m, q, sigma, alpha, **noise_kwargs)

        temp_m, temp_q, temp_sigma = m, q, sigma

        m, q, sigma = var_func(m_hat, q_hat, sigma_hat, reg_param)

        err = np.max(np.abs([(temp_m - m), (temp_q - q), (temp_sigma - sigma)]))

        m = blend * m + (1 - blend) * temp_m
        q = blend * q + (1 - blend) * temp_q
        sigma = blend * sigma + (1 - blend) * temp_sigma

    return m, q, sigma


def different_alpha_observables_fpeqs(
    var_func,
    var_hat_func,
    funs=[lambda m, q, sigma: 1 + q - 2 * m],
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    reg_param=0.1,
    initial_cond=[0.6, 0.0, 0.0],
    verbose=False,
    noise_kwargs={},
):
    n_observables = len(funs)
    alphas = np.logspace(
        np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points
    )
    out_values = np.empty((n_observables, n_alpha_points))

    for idx, alpha in enumerate(
        tqdm(alphas, desc="alpha", disable=not verbose, leave=False)
    ):
        m, q, sigma = state_equations(
            var_func,
            var_hat_func,
            reg_param=reg_param,
            alpha=alpha,
            init=initial_cond,
            noise_kwargs=noise_kwargs,
        )

        fixed_point_sols = {"m": m, "q": q, "sigma": sigma}
        for jdx, f in enumerate(funs):
            out_values[jdx, idx] = f(**fixed_point_sols)

    out_list = [out_values[idx, :] for idx in range(len(funs))]
    return alphas, out_list


# --------------------------


def var_func_BO(
    m_hat,
    q_hat,
    sigma_hat,
    reg_param,  # alpha, delta_small, delta_large, delta, percentage
):
    q = q_hat / (1 + q_hat)
    return q, q, 1 - q


def var_hat_func_BO_single_noise(m, q, sigma, alpha, delta):
    q_hat = alpha / (1 + delta - q)
    return q_hat, q_hat, q_hat


def var_hat_func_BO_num_single_noise(m, q, sigma, alpha, delta):
    q_hat = alpha * numfun.q_hat_equation_BO_single_noise(m, q, sigma, delta)
    return q_hat, q_hat, q_hat


def var_hat_func_BO_num_double_noise(
    m, q, sigma, alpha, delta_small, delta_large, percentage
):
    q_hat = alpha * numfun.q_hat_equation_BO_double_noise(
        m, q, sigma, delta_small, delta_large, percentage
    )
    return q_hat, q_hat, q_hat


# --------------------------


def var_func_L2(
    m_hat,
    q_hat,
    sigma_hat,
    reg_param,  # alpha, delta_small, delta_large, delta, percentage
):
    m = m_hat / (sigma_hat + reg_param)
    q = (np.square(m_hat) + q_hat) / np.square(sigma_hat + reg_param)
    sigma = 1.0 / (sigma_hat + reg_param)
    return m, q, sigma


def var_hat_func_L2_single_noise(m, q, sigma, alpha, delta):
    m_hat = alpha / (1 + sigma)
    q_hat = alpha * (1 + q + delta - 2 * np.abs(m)) / ((1 + sigma) ** 2)
    sigma_hat = alpha / (1 + sigma)
    return m_hat, q_hat, sigma_hat


def var_hat_func_L2_num_single_noise(m, q, sigma, alpha, delta):
    m_hat = alpha * numfun.m_hat_equation_L2_single_noise(m, q, sigma, delta)
    q_hat = alpha * numfun.q_hat_equation_L2_single_noise(m, q, sigma, delta)
    sigma_hat = -alpha * numfun.sigma_hat_equation_L2_single_noise(m, q, sigma, delta)
    return m_hat, q_hat, sigma_hat


def var_hat_func_L2_num_double_noise(
    m, q, sigma, alpha, delta_small, delta_large, percentage
):
    m_hat = alpha * numfun.m_hat_equation_L2_double_noise(
        m, q, sigma, delta_small, delta_large, percentage
    )
    q_hat = alpha * numfun.q_hat_equation_L2_double_noise(
        m, q, sigma, delta_small, delta_large, percentage
    )
    sigma_hat = -alpha * numfun.sigma_hat_equation_L2_double_noise(
        m, q, sigma, delta_small, delta_large, percentage
    )
    return m_hat, q_hat, sigma_hat


def var_hat_func_Huber_num_single_noise(m, q, sigma, alpha, delta, a):
    m_hat = alpha * numfun.m_hat_equation_Huber_single_noise(m, q, sigma, delta, a,)
    q_hat = alpha * numfun.q_hat_equation_Huber_single_noise(m, q, sigma, delta, a,)
    sigma_hat = -alpha * numfun.sigma_hat_equation_Huber_single_noise(
        m, q, sigma, delta, a,
    )
    return m_hat, q_hat, sigma_hat


def var_hat_func_Huber_num_double_noise(
    m, q, sigma, alpha, delta_small, delta_large, percentage, a
):
    m_hat = alpha * numfun.m_hat_equation_Huber_double_noise(
        m, q, sigma, delta_small, delta_large, percentage, a,
    )
    q_hat = alpha * numfun.q_hat_equation_Huber_double_noise(
        m, q, sigma, delta_small, delta_large, percentage, a,
    )
    sigma_hat = -alpha * numfun.sigma_hat_equation_Huber_double_noise(
        m, q, sigma, delta_small, delta_large, percentage, a,
    )
    return m_hat, q_hat, sigma_hat
