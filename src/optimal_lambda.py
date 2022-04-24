import numpy as np
from scipy.optimize import minimize, Bounds
from tqdm.auto import tqdm
import src.fpeqs as fp

SMALLEST_REG_PARAM = 1e-3
SMALLEST_HUBER_PARAM = 1e-3


def optimal_lambda(
    var_func,
    var_hat_func,
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    initial_cond=[0.6, 0.0, 0.0],
    verbose=False,
    var_hat_kwargs={},
    fun=lambda m, q, sigma: 1 + q - 2 * m,
):

    alphas = np.logspace(
        np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points
    )

    initial = initial_cond
    error_theory = np.zeros(n_alpha_points)
    reg_param_opt = np.zeros(n_alpha_points)

    for i, alpha in enumerate(
        tqdm(alphas, desc="alpha", disable=not verbose, leave=False)
    ):

        def error_func(reg_param):
            m, q, sigma = fp.state_equations(
                var_func,
                var_hat_func,
                reg_param=reg_param,
                alpha=alpha,
                init=initial,
                var_hat_kwargs=var_hat_kwargs,
            )
            return fun(**{"m": m, "q": q, "sigma": sigma})

        bnds = [(SMALLEST_REG_PARAM, None)]
        obj = minimize(error_func, x0=1.0, method="Nelder-Mead", bounds=bnds)
        if obj.success:
            error_theory[i] = obj.fun
            reg_param_opt[i] = obj.x
        else:
            raise RuntimeError("Minima could not be found.")

    return alphas, error_theory, reg_param_opt


def optimal_huber_parameter(
    double_noise=True,
    reg_param=1.5,
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    initial_cond=[0.6, 0.0, 0.0],
    verbose=False,
    noise_kwargs={},
    fun=lambda m, q, sigma: 1 + q - 2 * m,
):

    alphas = np.logspace(
        np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points
    )

    initial = initial_cond
    error_theory = np.zeros(n_alpha_points)
    a_opt = np.zeros(n_alpha_points)

    for i, alpha in enumerate(
        tqdm(alphas, desc="alpha", disable=not verbose, leave=False)
    ):

        def error_func(a):
            noise_kwargs.update({"a": a})
            m, q, sigma = fp.state_equations(
                fp.var_func_L2,
                fp.var_hat_func_Huber_num_double_noise
                if double_noise
                else fp.var_hat_func_Huber_num_single_noise,
                reg_param=reg_param,
                alpha=alpha,
                init=initial,
                noise_kwargs=noise_kwargs,
            )
            return fun(**{"m": m, "q": q, "sigma": sigma})

        obj = minimize(error_func, x0=1.0, method="Nelder-Mead")
        if obj.success:
            error_theory[i] = obj.fun
            a_opt[i] = obj.x
        else:
            raise RuntimeError("Minima could not be found.")

    return alphas, error_theory, a_opt


def optimal_reg_param_and_huber_parameter(
    double_noise=True,
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    initial_cond=[0.6, 0.0, 0.0],
    verbose=False,
    var_hat_kwargs={},
    fun=lambda m, q, sigma: 1 + q - 2 * m,
):
    alphas = np.logspace(
        np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points
    )

    initial = initial_cond
    error_theory = np.zeros(n_alpha_points)
    reg_param_opt = np.zeros(n_alpha_points)
    a_opt = np.zeros(n_alpha_points)

    for i, alpha in enumerate(
        tqdm(alphas, desc="alpha", disable=not verbose, leave=False)
    ):

        def error_func(x):
            reg_param, a = x
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

        bnds = [(SMALLEST_REG_PARAM, None), (SMALLEST_REG_PARAM, None)]
        obj = minimize(error_func, x0=[1.0, 1.0], method="Nelder-Mead", bounds=bnds)
        if obj.success:
            error_theory[i] = obj.fun
            reg_param_opt[i], a_opt[i] = obj.x
        else:
            raise RuntimeError("Minima could not be found.")

    return alphas, error_theory, reg_param_opt, a_opt
