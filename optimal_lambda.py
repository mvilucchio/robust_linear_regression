import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
from scipy.optimize import minimize
from tqdm.auto import tqdm
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
    verbose=False,
    eps=0.1,
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

        def error_func(lambd):
            m, q, _ = fixedpoint.state_equations(
                var_func,
                var_hat_func,
                delta_small=delta_small,
                delta_large=delta_large,
                lambd=lambd,
                alpha=alpha,
                eps=eps,
                init=initial,
            )
            return 1 + q - 2 * m

        obj = minimize(error_func, x0=0.5)
        error_theory[i] = obj.fun
        lambd_opt[i] = obj.x

    return alphas, error_theory, lambd_opt


if __name__ == "__main__":
    alpha_min, alpha_max = 0.01, 100
    eps = [0.01, 0.1, 0.3]
    alpha_points = 21
    deltas = [[1.0, 2.0], [1.0, 5.0], [1.0, 10.0]]

    alphas = [None] * len(deltas) * len(eps)
    errors = [None] * len(deltas) * len(eps)
    lambdas = [None] * len(deltas) * len(eps)

    colormap = fixedpoint.get_cmap(len(eps) * len(deltas) + 1)

    for idx, e in enumerate(tqdm(eps, desc="epsilon", leave=False)):
        for jdx, (delta_small, delta_large) in enumerate(
            tqdm(deltas, desc="delta", leave=False)
        ):
            while True:
                m = 0.89 * np.random.random() + 0.1
                q = 0.89 * np.random.random() + 0.1
                sigma = 0.89 * np.random.random() + 0.1
                if (
                    np.square(m) < q + delta_small * q
                    and np.square(m) < q + delta_large * q
                ):
                    break

            initial = [m, q, sigma]

            i = idx * len(deltas) + jdx

            alphas[i], errors[i], lambdas[i] = optimal_lambda(
                fixedpoint.var_func_L2,
                fixedpoint.var_hat_func_L2_num_eps,
                alpha_1=alpha_min,
                alpha_2=alpha_max,
                n_alpha_points=alpha_points,
                delta_small=delta_small,
                delta_large=delta_large,
                initial_cond=initial,
                verbose=True,
            )

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)

    for idx, e in enumerate(eps):
        for jdx, delta in enumerate(deltas):
            i = idx * len(deltas) + jdx
            ax.plot(
                alphas[i],
                errors[i],
                marker=".",
                label=r"$\epsilon = {}$ $\Delta = {}$".format(e, delta),
                color=colormap(i),
            )

    ax.set_title("L2 Loss - Double Noise")
    ax.set_ylabel(r"$\frac{1}{d} E[||\hat{w} - w^\star||^2]$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_on()
    ax.grid(True, which="both")
    ax.legend()

    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)

    for idx, e in enumerate(eps):
        for jdx, delta in enumerate(deltas):
            i = idx * len(deltas) + jdx
            ax.plot(
                alphas[i],
                lambdas[i],
                marker=".",
                label=r"$\epsilon = {}$ $\Delta = {}$".format(e, delta),
                color=colormap(i),
            )

    ax.set_title("L2 Loss - Double Noise")
    ax.set_ylabel(r"$\lambda_{opt}$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_on()
    ax.grid(True, which="both")
    ax.legend()
    plt.show()
