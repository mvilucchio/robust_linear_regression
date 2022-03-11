import numpy as np
from scipy.optimize import minimize
import fixed_point_equations_double as fixedpoint


def optimal_lambda(
    var_func,
    var_hat_func,
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    delta_small=1.0,
    delta_large=10.0,
    initial_cond=[0.6, 0.0, 0.0],
    eps=0.1,
):

    alphas = np.logspace(
        np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points
    )

    initial = initial_cond
    error_theory = np.zeros(n_alpha_points)
    lambd_opt = np.zeros(n_alpha_points)

    # for to parallelize is this one. By parallelizing in this file should already take
    # care of all the calls
    for i, alpha in enumerate(alphas):

        def error_func(reg_param):
            m, q, _ = fixedpoint.state_equations(
                var_func,
                var_hat_func,
                delta_small=delta_small,
                delta_large=delta_large,
                lambd=reg_param,
                alpha=alpha,
                eps=eps,
                init=initial,
            )
            return 1 + q - 2 * m

        obj = minimize(error_func, x0=1.0, method="Nelder-Mead")
        if obj.success:
            error_theory[i] = obj.fun
            lambd_opt[i] = obj.x
        else:
            raise RuntimeError("Minima could not be found")

    return alphas, error_theory, lambd_opt
