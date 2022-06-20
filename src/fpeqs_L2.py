import src.numerical_functions as numfun
from numba import njit
import numpy as np


@njit(error_model="numpy", fastmath=True)
def var_func_L2(
    m_hat, q_hat, sigma_hat, reg_param,
):
    m = m_hat / (sigma_hat + reg_param)
    q = (m_hat ** 2 + q_hat) / (sigma_hat + reg_param) ** 2
    sigma = 1.0 / (sigma_hat + reg_param)
    return m, q, sigma


@njit(error_model="numpy", fastmath=True)
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


@njit(error_model="numpy", fastmath=True)
def var_hat_func_L2_double_noise(
    m, q, sigma, alpha, delta_small, delta_large, percentage
):
    delta_eff = (1 - percentage) * delta_small + percentage * delta_large
    m_hat = alpha / (1 + sigma)
    q_hat = alpha * (1 + q + delta_eff - 2 * np.abs(m)) / ((1 + sigma) ** 2)
    sigma_hat = alpha / (1 + sigma)
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


@njit(error_model="numpy", fastmath=True)
def var_hat_func_L2_decorrelated_noise(
    m, q, sigma, alpha, delta_small, delta_large, percentage, beta
):
    delta_eff = (1 - percentage) * delta_small + percentage * delta_large
    intermediate_val = 1 + percentage * (beta - 1)

    m_hat = alpha * intermediate_val / (1 + sigma)
    q_hat = (
        alpha
        * (
            1
            + q
            + delta_eff
            + percentage * (beta ** 2 - 1)
            - 2 * np.abs(m) * intermediate_val
        )
        / ((1 + sigma) ** 2)
    )
    sigma_hat = alpha / (1 + sigma)
    return m_hat, q_hat, sigma_hat


def var_hat_func_L2_num_decorrelated_noise(
    m, q, sigma, alpha, delta_small, delta_large, percentage, beta
):
    m_hat = alpha * numfun.m_hat_equation_L2_decorrelated_noise(
        m, q, sigma, delta_small, delta_large, percentage, beta
    )
    q_hat = alpha * numfun.q_hat_equation_L2_decorrelated_noise(
        m, q, sigma, delta_small, delta_large, percentage, beta
    )
    sigma_hat = -alpha * numfun.sigma_hat_equation_L2_decorrelated_noise(
        m, q, sigma, delta_small, delta_large, percentage, beta
    )
    return m_hat, q_hat, sigma_hat
