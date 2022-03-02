import numpy as np
import matplotlib.pyplot as plt
from optimal_lambda import optimal_lambda
from tqdm.auto import tqdm
import fixed_point_equations_double as fixedpoint
import fixed_point_equations_bayes_opt as bofpe
import os

names_cm = ["Blues", "Reds", "Greens", "Oranges"]

if __name__ == "__main__":
    random_number = np.random.randint(0, 100)

    alpha_min, alpha_max = 0.01, 100
    eps = [0.01, 0.1, 0.3]
    alpha_points = 21
    deltas = [[0.5, 1.5]]

    alphas = [None] * len(deltas) * len(eps)
    errors = [None] * len(deltas) * len(eps)
    lambdas = [None] * len(deltas) * len(eps)

    colormap = fixedpoint.get_cmap(len(eps) * len(deltas) + 1)

    # evaluates the lambda optimal for each value
    for idx, e in enumerate(tqdm(eps, desc="epsilon", leave=False)):
        for jdx, (delta_small, delta_large) in enumerate(
            tqdm(deltas, desc="delta", leave=False)
        ):
            i = idx * len(deltas) + jdx
            # print(
            #     "lambda_opt - eps {} - deltas {} deltal {}.npz".format(
            #         e, delta_small, delta_large
            #     )
            # )
            file_exists = os.path.exists(
                "lambda_opt - eps {} - deltas {} deltal {}.npz".format(
                    e, delta_small, delta_large
                )
            )

            if not file_exists:
                while True:
                    m = 0.89 * np.random.random() + 0.1
                    q = 0.89 * np.random.random() + 0.1
                    sigma = np.random.random()
                    if (
                        np.square(m) < q + delta_small * q
                        and np.square(m) < q + delta_large * q
                    ):
                        break

                initial = [m, q, sigma]

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

                np.savez(
                    "lambda_opt - eps {} - deltas {} deltal {}".format(
                        e, delta_small, delta_large
                    ),
                    alphas=alphas[i],
                    errors=errors[i],
                    lambdas=lambdas[i],
                )

            else:
                data = np.load(
                    "lambda_opt - eps {} - deltas {} deltal {}.npz".format(
                        e, delta_small, delta_large
                    )
                )

                alphas[i] = data["alphas"]
                errors[i] = data["errors"]
                lambdas[i] = data["lambdas"]

    # evaluates the bayes optimal

    alphasBO = [None] * len(deltas) * len(eps)
    errorsBO = [None] * len(deltas) * len(eps)

    for idx, e in enumerate(tqdm(eps, desc="lambda", leave=False)):
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

            # alphasBO[i], errorsBO[i] = fixedpoint.projection_ridge_different_alpha_theory(
            #     fixedpoint.var_func_BO,
            #     fixedpoint.var_hat_func_BO_num_eps,
            #     alpha_1=alpha_min,
            #     alpha_2=alpha_max,
            #     n_alpha_points=alpha_points,
            #     lambd=1.0,
            #     delta_small=delta_small,
            #     delta_large=delta_large,
            #     initial_cond=initial,
            #     verbose=True,
            #     eps=e,
            # )

            alphasBO[i], errorsBO[i] = bofpe.bayes_opt_theory(
                bofpe.var_hat_func_bayes_opt_eps,
                alpha_1=alpha_min,
                alpha_2=alpha_max,
                n_alpha_points=alpha_points,
                delta_small=delta_small,
                delta_large=delta_large,
                initial_cond=initial,
                verbose=True,
            )

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)

    colormap_BO = fixedpoint.get_cmap(len(eps) * len(deltas) + 2, name="Greys")

    colormap = fixedpoint.get_cmap(len(eps) * len(deltas) + 2, name="Reds")

    for idx, e in enumerate(eps):
        for jdx, delta in enumerate(deltas):
            i = idx * len(deltas) + jdx

            ax.plot(
                alphas[i],
                errors[i],
                marker=".",
                label=r"$\epsilon = {}$ $\Delta = {}$".format(e, delta),
                color=colormap(i + 2),
            )

    for idx, e in enumerate(eps):
        for jdx, delta in enumerate(deltas):
            i = idx * len(deltas) + jdx
            ax.plot(
                alphasBO[i],
                errorsBO[i],
                marker=".",
                label=r"$BO \epsilon = {}$ $\Delta = {}$".format(e, delta),
                color=colormap_BO(i + 2),
            )

    ax.set_title(r"L2 confront BO vs $\lambda_{opt}$")
    ax.set_ylabel(r"$\frac{1}{d} E[||\hat{w} - w^\star||^2]$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_on()
    ax.grid(True, which="both")
    ax.legend()

    fig.savefig("./imgs/BOvslambdaOptL2_both_{}.png".format(random_number), format="png")

    for idx, e in enumerate(eps):
        cmap = fixedpoint.get_cmap(len(deltas) + 1, name=names_cm[idx])
        for jdx, delta in enumerate(deltas):
            #  i = idx * len(deltas) + jdx
            ax2.plot(
                alphas[i],
                errors[i] - errorsBO[i],
                marker=".",
                color=cmap(jdx + 1),
                label=r"$\epsilon = {}$ $\Delta = {}$".format(e, delta),
            )

    ax2.set_title(r"L2 confront BO vs $\lambda_{opt}$")
    ax2.set_ylabel(
        r"$\frac{1}{d} E[||\hat{w} - w^\star||^2]_{\lambda_{opt}} - \frac{1}{d} E[||\hat{w} - w^\star||^2]_{BO}$"
    )
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_xscale("log")
    # ax2.set_yscale("log")
    ax2.minorticks_on()
    ax2.grid(True, which="both")
    ax2.legend()

    fig2.savefig(
        "./imgs/BOvslambdaOptL2_difference_{}.png".format(random_number), format="png"
    )

    for idx, e in enumerate(eps):
        cmap = fixedpoint.get_cmap(len(deltas) + 1, name=names_cm[idx])
        for jdx, delta in enumerate(deltas):
            #  i = idx * len(deltas) + jdx
            ax3.plot(
                alphas[i],
                (errors[i] - errorsBO[i]) / errorsBO[i],
                marker=".",
                color=cmap(jdx + 1),
                label=r"$\epsilon = {}$ $\Delta = {}$".format(e, delta),
            )

    ax3.set_title(r"L2 confront BO vs $\lambda_{opt}$")
    ax3.set_ylabel(
        r"$(E[||\hat{w} - w^\star||^2]_{\lambda_{opt}} - E[||\hat{w} - w^\star||^2]_{BO} ) / E[||\hat{w} - w^\star||^2]_{BO}$"
    )
    ax3.set_xlabel(r"$\alpha$")
    ax3.set_xscale("log")
    # ax3.set_yscale("log")
    ax3.minorticks_on()
    ax3.grid(True, which="both")
    ax3.legend()

    fig3.savefig(
        "./imgs/BOvslambdaoOtL2_division_{}.png".format(random_number), format="png"
    )

    plt.show()
