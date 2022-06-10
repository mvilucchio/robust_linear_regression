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

# y is fixedto zero
@vectorize
def integrate_fun_outliers(z, V, omega, delta, mu):
    if np.abs(np.sqrt(V) * z + omega) > delta:
        return (
            0.5
            # / np.sqrt(2 * np.pi)
            # * np.exp(-0.5 * z ** 2)
            * (mu / (1 + mu))
            * (delta ** mu)
            * np.abs(1 / (np.sqrt(V) * z + omega)) ** (mu + 1.0)
        )
    else:
        return (
            0.5 * mu / ((1 + mu) * delta)  # / np.sqrt(2 * np.pi) * np.exp(-0.5 * z ** 2)
        )  #


# @vectorize
# def integrate_fun(z, V, omega, delta):
#     if np.abs(-np.sqrt(V) * z - omega) <= delta:
#         return np.exp(-(z ** 2) / 2) * np.exp(
#             -((-np.sqrt(V) * z - omega) ** 2) / (2 * delta)
#         )
#     else:
#         return np.exp(-(z ** 2) / 2) * np.exp(-np.abs(-np.sqrt(V) * z - omega) / 2)


def minimize_fun(omega, x, V, eps, delta, mu):
    return (x - omega) ** 2 / (2 * V) + np.log(
        (1 - eps)
        / np.sqrt(2 * np.pi * (delta + V))
        * np.exp(-0.5 * omega ** 2 / (delta + V))
        + eps * quad(integrate_fun_outliers, -500, 500, args=(V, omega, delta, mu))[0]
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


def _minimize_my(x, V, eps, delta, mu):
    res = minimize(minimize_fun, x0=np.abs(x), args=(x, V, eps, delta, mu), tol=1e-1,)
    if res.success:
        return -res.fun
    else:
        raise RuntimeError("Minima could not be found.")


if __name__ == "__main__":

    eps, V, mu = 0.3, 0.01, 1.5
    mus = [0.5, 1.5, 2.5]
    epses = [0.01, 0.05, 0.1, 0.3]
    deltas = [0.5, 1.0, 2.0, 5.0, 10.0]
    delta = 5.0

    xs = np.linspace(-30, 30, 300)

    # loss_values = np.empty((len(epses), len(xs)))
    loss_values = np.empty((len(deltas), len(xs)))

    for jdx, d in enumerate(deltas):

        inputs = [(x, V, eps, d, mu) for x in xs]

        with Pool() as pool:
            results = pool.starmap(_minimize_my, inputs)

        for idx, l in enumerate(results):
            loss_values[jdx][idx] = l

    for idx, d in enumerate(deltas):
        plt.plot(xs, loss_values[jdx], label=r"$\Delta$ = {:.1f}".format(d))

    # plt.yscale("log")
    # plt.xscale("log")
    plt.legend()
    plt.show()

    # xs = np.linspace(-15, 15, 2000)
    # for m in [0.5, 1.5, 2.5, 5.0, 10.0]:
    #     plt.plot(
    #         xs,
    #         integrate_fun_outliers(xs, 1, 0, 3.5, m),
    #         label=r"$\mu = {:.2f}$".format(m),
    #     )
    # plt.legend()
    # plt.grid()

    # plt.show()
