import numpy as np
from numba import njit
import src.numerical_functions as numfun

from multiprocessing import Pool
from tqdm.auto import tqdm

# from mpi4py import MPI
# from mpi4py.futures import MPIPoolExecutor as Pool

BLEND = 0.5
TOL_FPE = 5e-8


def state_equations(
    var_func,
    var_hat_func,
    reg_param=0.01,
    alpha=0.5,
    init=(0.5, 0.5, 0.5),
    var_hat_kwargs={"delta_small": 0.1, "delta_large": 2.0, "percentage": 0.1},
):
    m, q, sigma = init[0], init[1], init[2]
    err = 1.0
    blend = BLEND
    while err > TOL_FPE:
        m_hat, q_hat, sigma_hat = var_hat_func(m, q, sigma, alpha, **var_hat_kwargs)

        temp_m, temp_q, temp_sigma = m, q, sigma

        m, q, sigma = var_func(m_hat, q_hat, sigma_hat, reg_param)

        err = np.max(np.abs([(temp_m - m), (temp_q - q), (temp_sigma - sigma)]))

        m = blend * m + (1 - blend) * temp_m
        q = blend * q + (1 - blend) * temp_q
        sigma = blend * sigma + (1 - blend) * temp_sigma

        # print(
        #     "     err: {:.10f} alpha : {:.2f} m : {:.7f} q : {:.7f} sigma : {:.7f} a : {:.7f} reg_par : {:.7f}".format(
        #         err, alpha, m, q, sigma, var_hat_kwargs["a"], reg_param
        #     )
        # )

    return m, q, sigma


def _find_fixed_point(
    alpha, var_func, var_hat_func, reg_param, initial_cond, var_hat_kwargs
):
    m, q, sigma = state_equations(
        var_func,
        var_hat_func,
        reg_param=reg_param,
        alpha=alpha,
        init=initial_cond,
        var_hat_kwargs=var_hat_kwargs,
    )
    return m, q, sigma


def no_parallel_different_alpha_observables_fpeqs(
    var_func,
    var_hat_func,
    funs=[lambda m, q, sigma: 1 + q - 2 * m],
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    reg_param=0.1,
    initial_cond=[0.6, 0.0, 0.0],
    var_hat_kwargs={},
):
    n_observables = len(funs)
    alphas = np.logspace(
        np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points
    )
    out_values = np.empty((n_observables, n_alpha_points))
    results = [None] * len(alphas)

    for idx, a in enumerate(tqdm(alphas)):
        results[idx] = _find_fixed_point(
            a, var_func, var_hat_func, reg_param, initial_cond, var_hat_kwargs
        )
    # inputs = [
    #     (a, var_func, var_hat_func, reg_param, initial_cond, var_hat_kwargs)
    #     for a in alphas
    # ]

    # with Pool() as pool:
    #     results = pool.starmap(_find_fixed_point, inputs)

    for idx, (m, q, sigma) in enumerate(results):
        fixed_point_sols = {"m": m, "q": q, "sigma": sigma}
        for jdx, f in enumerate(funs):
            out_values[jdx, idx] = f(**fixed_point_sols)

    out_list = [out_values[idx, :] for idx in range(len(funs))]
    return alphas, out_list


def MPI_different_alpha_observables_fpeqs(
    var_func,
    var_hat_func,
    funs=[lambda m, q, sigma: 1 + q - 2 * m],
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    reg_param=0.1,
    initial_cond=[0.6, 0.0, 0.0],
    var_hat_kwargs={},
):
    comm = MPI.COMM_WORLD
    i = comm.Get_rank()
    pool_size = comm.Get_size()

    n_observables = len(funs)
    alphas = np.logspace(
        np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), pool_size
    )
    alpha = alphas[i]
    out_values = np.empty((n_observables, n_alpha_points))

    m, q, sigma = _find_fixed_point(
        alpha, var_func, var_hat_func, reg_param, initial_cond, var_hat_kwargs
    )

    ms = np.empty(pool_size)
    qs = np.empty(pool_size)
    sigmas = np.empty(pool_size)

    if i == 0:
        ms[0] = m
        qs[0] = q
        sigmas[0] = sigma

        for j in range(1, pool_size):
            ms[j] = comm.recv(source=j)
        for j in range(1, pool_size):
            qs[j] = comm.recv(source=j)
        for j in range(1, pool_size):
            sigmas[j] = comm.recv(source=j)

        for idx, (mm, qq, ssigma) in enumerate(zip(ms, qs, sigmas)):
            fixed_point_sols = {"m": mm, "q": qq, "sigma": ssigma}
            for jdx, f in enumerate(funs):
                out_values[jdx, idx] = f(**fixed_point_sols)

        out_list = [out_values[idx, :] for idx in range(len(funs))]
        return alphas, out_list
    else:
        print("Process {} sending {}".format(i, reg_param))
        comm.send(m, dest=0)
        comm.send(q, dest=0)
        comm.send(sigma, dest=0)


def different_alpha_observables_fpeqs(
    var_func,
    var_hat_func,
    funs=[lambda m, q, sigma: 1 + q - 2 * m],
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    reg_param=0.1,
    initial_cond=[0.6, 0.0, 0.0],
    var_hat_kwargs={},
):
    n_observables = len(funs)
    alphas = np.logspace(
        np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points
    )
    out_values = np.empty((n_observables, n_alpha_points))

    inputs = [
        (a, var_func, var_hat_func, reg_param, initial_cond, var_hat_kwargs)
        for a in alphas
    ]

    with Pool() as pool:
        results = pool.starmap(_find_fixed_point, inputs)

    for idx, (m, q, sigma) in enumerate(results):
        fixed_point_sols = {"m": m, "q": q, "sigma": sigma}
        for jdx, f in enumerate(funs):
            out_values[jdx, idx] = f(**fixed_point_sols)

    out_list = [out_values[idx, :] for idx in range(len(funs))]
    return alphas, out_list


def different_reg_param_gen_error(
    var_func,
    var_hat_func,
    funs=[lambda m, q, sigma: 1 + q - 2 * m],
    reg_param_1=0.01,
    reg_param_2=100,
    n_reg_param_points=16,
    alpha=0.1,
    initial_cond=[0.6, 0.0, 0.0],
    var_hat_kwargs={},
):
    n_observables = len(funs)
    reg_params = np.logspace(
        np.log(reg_param_1) / np.log(10),
        np.log(reg_param_2) / np.log(10),
        n_reg_param_points,
    )
    out_values = np.empty((n_observables, n_reg_param_points))

    inputs = [
        (alpha, var_func, var_hat_func, rp, initial_cond, var_hat_kwargs)
        for rp in reg_params
    ]

    with Pool() as pool:
        results = pool.starmap(_find_fixed_point, inputs)

    for idx, (m, q, sigma) in enumerate(results):
        fixed_point_sols = {"m": m, "q": q, "sigma": sigma}
        for jdx, f in enumerate(funs):
            out_values[jdx, idx] = f(**fixed_point_sols)

    out_list = [out_values[idx, :] for idx in range(len(funs))]
    return reg_params, out_list


# --------------------------


@njit(error_model="numpy", fastmath=True)
def var_func_BO(
    m_hat, q_hat, sigma_hat, reg_param,
):
    q = q_hat / (1 + q_hat)
    return q, q, 1 - q


@njit(error_model="numpy", fastmath=True)
def var_hat_func_BO_single_noise(m, q, sigma, alpha, delta):
    q_hat = alpha / (1 + delta - q)
    return q_hat, q_hat, q_hat


def var_hat_func_BO_num_single_noise(m, q, sigma, alpha, delta):
    q_hat = alpha * numfun.q_hat_equation_BO_single_noise(m, q, sigma, delta)
    return q_hat, q_hat, q_hat


def var_hat_func_BO_double_noise(
    m, q, sigma, alpha, delta_small, delta_large, percentage
):
    raise NotImplementedError
    q_hat = (
        alpha
        * (1 + (1 - percentage) * delta_large + percentage * delta_small - q)
        / ((1 + delta_small - q) * (1 + delta_large - q))
    )
    return q_hat, q_hat, q_hat


def var_hat_func_BO_num_double_noise(
    m, q, sigma, alpha, delta_small, delta_large, percentage
):
    q_hat = alpha * numfun.q_hat_equation_BO_double_noise(
        m, q, sigma, delta_small, delta_large, percentage
    )
    return q_hat, q_hat, q_hat


def var_hat_func_BO_decorrelated_noise(
    m, q, sigma, alpha, delta_small, delta_large, percentage, beta
):
    raise NotImplementedError
    q_hat = (
        alpha
        * (1 + (1 - percentage) * delta_large + percentage * delta_small - q)
        / ((1 + delta_small - q) * (1 + delta_large - q))
    )
    return q_hat, q_hat, q_hat


def var_hat_func_BO_num_decorrelated_noise(
    m, q, sigma, alpha, delta_small, delta_large, percentage, beta
):
    q_hat = alpha * numfun.q_hat_equation_BO_decorrelated_noise(
        m, q, sigma, delta_small, delta_large, percentage, beta
    )
    return q_hat, q_hat, q_hat


# --------------------------


@njit(error_model="numpy", fastmath=True)
def var_func_L2(
    m_hat, q_hat, sigma_hat, reg_param,
):
    m = m_hat / (sigma_hat + reg_param)
    q = (np.square(m_hat) + q_hat) / np.square(sigma_hat + reg_param)
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


def var_hat_func_Huber_num_single_noise(m, q, sigma, alpha, delta, a):
    m_hat = alpha * numfun.m_hat_equation_Huber_single_noise(m, q, sigma, delta, a)
    q_hat = alpha * numfun.q_hat_equation_Huber_single_noise(m, q, sigma, delta, a)
    sigma_hat = -alpha * numfun.sigma_hat_equation_Huber_single_noise(
        m, q, sigma, delta, a
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


def var_hat_func_Huber_num_decorrelated_noise(
    m, q, sigma, alpha, delta_small, delta_large, percentage, beta, a
):
    m_hat = alpha * numfun.m_hat_equation_Huber_decorrelated_noise(
        m, q, sigma, delta_small, delta_large, percentage, beta, a,
    )
    q_hat = alpha * numfun.q_hat_equation_Huber_decorrelated_noise(
        m, q, sigma, delta_small, delta_large, percentage, beta, a,
    )
    sigma_hat = -alpha * numfun.sigma_hat_equation_Huber_decorrelated_noise(
        m, q, sigma, delta_small, delta_large, percentage, beta, a,
    )
    return m_hat, q_hat, sigma_hat


def var_hat_func_numerical_loss_single_noise(
    m, q, sigma, alpha, delta, precompute_proximal_func, loss_args
):
    m_int, q_int, sigma_int = numfun.hat_equations_numerical_loss_single_noise(
        m, q, sigma, delta, precompute_proximal_func, loss_args
    )
    # print("m_int {} q_int {} sigma_int {}".format(m_int, q_int, sigma_int))
    m_hat = alpha * m_int
    q_hat = alpha * q_int
    sigma_hat = -alpha * sigma_int
    return m_hat, q_hat, sigma_hat


def var_hat_func_numerical_loss_double_noise(
    m,
    q,
    sigma,
    alpha,
    delta_small,
    delta_large,
    percentage,
    loss_derivative,
    loss_second_derivative,
    loss_args=None,
):
    m_int, q_int, sigma_int = numfun.hat_equations_numerical_loss_double_noise(
        m,
        q,
        sigma,
        delta_small,
        delta_large,
        percentage,
        loss_derivative,
        loss_second_derivative,
        loss_args=loss_args,
    )
    m_hat = alpha * m_int
    q_hat = alpha * q_int
    sigma_hat = -alpha * sigma_int
    return m_hat, q_hat, sigma_hat


def var_hat_func_numerical_loss_decorrelated_noise(
    m,
    q,
    sigma,
    alpha,
    delta_small,
    delta_large,
    percentage,
    beta,
    loss_derivative,
    loss_second_derivative,
    loss_args=None,
):
    m_int, q_int, sigma_int = numfun.hat_equations_numerical_loss_decorrelated_noise(
        m,
        q,
        sigma,
        delta_small,
        delta_large,
        percentage,
        beta,
        loss_derivative,
        loss_second_derivative,
        loss_args=loss_args,
    )
    m_hat = alpha * m_int
    q_hat = alpha * q_int
    sigma_hat = -alpha * sigma_int
    return m_hat, q_hat, sigma_hat
