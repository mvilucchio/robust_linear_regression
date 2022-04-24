import numpy as np
from scipy.optimize import minimize, Bounds
import src_cluster.fpeqs_cluster_version as fp
from multiprocessing import Pool

SMALLEST_REG_PARAM = 1e-3
SMALLEST_HUBER_PARAM = 1e-3


def _find_optimal_reg_param_gen_error(
    alpha, var_func, var_hat_func, initial, var_hat_kwargs
):
    def minimize_fun(reg_param):
        m, q, _ = fp.state_equations(
            var_func,
            var_hat_func,
            reg_param=reg_param,
            alpha=alpha,
            init=initial,
            var_hat_kwargs=var_hat_kwargs,
        )
        return 1 + q - 2 * m

    bnds = [(SMALLEST_REG_PARAM, None)]
    obj = minimize(minimize_fun, x0=1.0, method="Nelder-Mead", bounds=bnds)
    if obj.success:
        fun_val = obj.fun
        reg_param_opt = obj.x
        return fun_val, reg_param_opt
    else:
        raise RuntimeError("Minima could not be found.")


def optimal_lambda(
    var_func,
    var_hat_func,
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    initial_cond=[0.6, 0.0, 0.0],
    var_hat_kwargs={},
):
    alphas = np.logspace(
        np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points
    )

    initial = initial_cond
    fun_values = np.zeros(n_alpha_points)
    reg_param_opt = np.zeros(n_alpha_points)

    inputs = [(a, var_func, var_hat_func, initial, var_hat_kwargs) for a in alphas]

    with Pool() as pool:
        results = pool.starmap(_find_optimal_reg_param_gen_error, inputs)

    for idx, (e, regp) in enumerate(results):
        fun_values[idx] = e
        reg_param_opt[idx] = regp

    return alphas, fun_values, reg_param_opt


def _find_optimal_huber_parameter_gen_error(
    alpha, double_noise, reg_param, initial, var_hat_kwargs, fun
):
    def error_func(a):
        var_hat_kwargs.update({"a": a})
        m, q, sigma = fp.state_equations(
            fp.var_func_L2,
            fp.var_hat_func_Huber_num_double_noise
            if double_noise
            else fp.var_hat_func_Huber_num_single_noise,
            reg_param=reg_param,
            alpha=alpha,
            init=initial,
            var_hat_kwargs=var_hat_kwargs,
        )
        return fun(**{"m": m, "q": q, "sigma": sigma})

    obj = minimize(error_func, x0=1.0, method="Nelder-Mead")
    if obj.success:
        fun_val = obj.fun
        a_opt = obj.x
        return fun_val, a_opt
    else:
        raise RuntimeError("Minima could not be found.")


def optimal_huber_parameter(
    double_noise=True,
    reg_param=1.5,
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    initial_cond=[0.6, 0.0, 0.0],
    var_hat_kwargs={},
    fun=lambda m, q, sigma: 1 + q - 2 * m,
):

    alphas = np.logspace(
        np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points
    )

    initial = initial_cond
    fun_values = np.zeros(n_alpha_points)
    a_opt = np.zeros(n_alpha_points)

    with Pool() as pool:
        results = pool.map(
            lambda a: _find_optimal_huber_parameter_gen_error(
                a, double_noise, reg_param, initial, var_hat_kwargs, fun
            ),
            alphas,
        )

    for idx, (e, a) in enumerate(results):
        fun_values[idx] = e
        a_opt[idx] = a

    return alphas, fun_values, a_opt


def _find_optimal_reg_param_and_huber_parameter_gen_error(
    alpha, double_noise, initial, var_hat_kwargs
):
    def minimize_fun(x):
        reg_param, a = x
        var_hat_kwargs.update({"a": a})
        m, q, _ = fp.state_equations(
            fp.var_func_L2,
            fp.var_hat_func_Huber_num_double_noise
            if double_noise
            else fp.var_hat_func_Huber_num_single_noise,
            reg_param=reg_param,
            alpha=alpha,
            init=initial,
            var_hat_kwargs=var_hat_kwargs,
        )
        return 1 + q - 2 * m

    bnds = [(SMALLEST_REG_PARAM, None), (SMALLEST_REG_PARAM, None)]
    obj = minimize(minimize_fun, x0=[1.0, 1.0], method="Nelder-Mead", bounds=bnds)
    if obj.success:
        fun_val = obj.fun
        reg_param_opt, a_opt = obj.x
        return fun_val, reg_param_opt, a_opt
    else:
        raise RuntimeError("Minima could not be found.")


def optimal_reg_param_and_huber_parameter(
    double_noise=True,
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    initial_cond=[0.6, 0.0, 0.0],
    var_hat_kwargs={},
):
    alphas = np.logspace(
        np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points
    )

    initial = initial_cond
    fun_values = np.zeros(n_alpha_points)
    reg_param_opt = np.zeros(n_alpha_points)
    a_opt = np.zeros(n_alpha_points)

    inputs = [(a, double_noise, initial, var_hat_kwargs) for a in alphas]

    with Pool() as pool:
        results = pool.starmap(_find_optimal_reg_param_and_huber_parameter_gen_error, inputs)

    for idx, (e, regp, a) in enumerate(results):
        fun_values[idx] = e
        reg_param_opt[idx] = regp
        a_opt[idx] = a

    return alphas, fun_values, reg_param_opt, a_opt
