import numpy as np
from scipy.optimize import minimize
from tqdm.auto import tqdm
import src.fpeqs as fp


def optimal_lambda(
    var_func,
    var_hat_func,
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
    lambd_opt = np.zeros(n_alpha_points)

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
                noise_kwargs=noise_kwargs,
            )
            return fun(**{"m": m, "q": q, "sigma": sigma})

        obj = minimize(error_func, x0=1.0, method="Nelder-Mead")
        if obj.success:
            error_theory[i] = obj.fun
            lambd_opt[i] = obj.x
        else:
            raise RuntimeError("Minima could not be found")

    return alphas, error_theory, lambd_opt
