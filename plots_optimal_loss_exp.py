from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from src.utils import load_file
import src.plotting_utils as pu
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from itertools import product
from numba import njit, vectorize
from multiprocessing import Pool
from tqdm.auto import tqdm
from src.utils import load_file


@vectorize
def integrate_fun(z, V, omega, beta, delta_large, a, alpha):
    if :
        return np.exp(-(z ** 2) / 2) * 
    elif :
        return np.exp(-(z ** 2) / 2) * 
    else:
        return np.exp(-(z ** 2) / 2) * np.exp()


# @vectorize
# def integrate_fun(z, V, omega, delta):
#     if np.abs(-np.sqrt(V) * z - omega) <= delta:
#         return np.exp(-(z ** 2) / 2) * np.exp(
#             -((-np.sqrt(V) * z - omega) ** 2) / (2 * delta)
#         )
#     else:
#         return np.exp(-(z ** 2) / 2) * np.exp(-np.abs(-np.sqrt(V) * z - omega) / 2)


def minimize_fun(omega, x, V, eps, delta_small, delta_large, beta):
    return (
        (x - omega) ** 2 / (2 * V)
        + np.log(
            quad(integrate_fun, -100, 100, args=(V, omega, beta, delta_large))
        )
    )


# def minimize_fun(omega, x, V, eps, delta_small, delta_large, beta):
#     return (x - omega) ** 2 / (2 * V) + np.log(
#         (1 - eps)
#         * np.exp(
#             -((-omega) ** 2) / (2 * (V + delta_small))
#             + ((-beta * omega) ** 2) / (2 * (beta ** 2 * V + delta_large))
#         )
#         / np.sqrt(2 * np.pi * (V + delta_small))
#         + eps / np.sqrt(2 * np.pi * (beta ** 2 * V + delta_large))
#     )


def _minimize_my(x, V, eps, delta_small, delta_large, beta):
    res = minimize(
        minimize_fun,
        x0=np.abs(x),
        args=(x, V, eps, delta_small, delta_large, beta),
        tol=1e-1,
    )
    if res.success:
        return -res.fun
    else:
        raise RuntimeError("Minima could not be found.")


if __name__ == "__main__":

    percentage, delta_small, delta_large = 0.1, 0.1, 5.0
    eps, beta = percentage, 1.0
    delta_large = [0.5, 1.0, 2.0, 5.0, 10.0]

    experiments_settings = [
        {
            # "loss_name": "L2",
            "alpha_min": 0.01,
            "alpha_max": 100,
            "alpha_pts": 20,
            # "alpha_pts_theoretical": 36,
            # "alpha_pts_experimental": 4,
            # "reg_param": 1.0,
            # "delta": 0.5,
            "delta_small": delta_small,
            "delta_large": dl,
            "percentage": percentage,
            # "n_features": 500,
            #  "repetitions": 4,
            "beta": beta,
            "experiment_type": "BO",
        }
        for dl in delta_large
    ]

    for exp_d, dl in zip(tqdm(experiments_settings), delta_large):
        alphas, Vs = load_file(**exp_d)

        for a, V in zip(alphas, Vs):

            xs = np.linspace(-30, 30, 300)
            loss_values = np.empty_like(xs)

            inputs = [(x, V, eps, delta_small, dl, beta) for x in xs]

            with Pool() as pool:
                results = pool.starmap(_minimize_my, inputs)

            for idx, l in enumerate(results):
                loss_values[idx] = l

            np.savez(
                "ass_optimal_loss_double_gauss_{}_{}_{}_{}_{}".format(
                    eps, delta_small, dl, beta, a
                ),
                x=xs,
                loss=loss_values,
            )
