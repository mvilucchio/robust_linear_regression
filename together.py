import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
from tqdm.auto import tqdm
import fixed_point_equations as fpe
import numerics as num
import os

theory_path = "./data/theory"
experiments_path = "./data/experiments"

data_dir_exists = os.path.exists("./data")
if not data_dir_exists:
    os.makedirs("./data")
    os.makedirs("./data/theory")
    os.makedirs("./data/experiments")

theory_dir_exists = os.path.exists("./data/theory")
experiments_dir_exists = os.path.exists("./data/experiments")

if not theory_dir_exists:
    os.makedirs("./data/theory")

if not experiments_dir_exists:
    os.makedirs("./data/experiments")

random_number = np.random.randint(0, 100)

names_cm = ["Purples", "Blues", "Greens", "Oranges", "Greys"]


def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def file_name_generator(
    loss_name,
    alpha_min,
    alpha_max,
    alpha_points,
    dim,
    reps,
    delta_small,
    delta_large,
    lamb,
    eps,
    experiments=True,
    double_noise=True,
):
    if experiments:
        return "{} - eps {} - exp - alphas [{} {} {:d}] - dim {:d} - rep {:d} - delta [{} {}] - lambda {}".format(
            loss_name,
            eps,
            alpha_min,
            alpha_max,
            alpha_points,
            dim,
            reps,
            delta_small,
            delta_large,
            lamb,
        )
    else:
        return "{} - eps {} - theory - alphas [{} {} {:d}] - rep {:d} - delta [{} {}] - lambda {}".format(
            loss_name,
            eps,
            alpha_min,
            alpha_max,
            alpha_points,
            reps,
            delta_small,
            delta_large,
            lamb,
        )


loss_chosen = "L2"
eps = 0.0
alpha_min, alpha_max = 0.01, 100
alpha_points_theory = 51
alpha_points_num = 31
d = 500
reps = 10
deltas = [[1.0, 0.0]]
lambdas = [0.01, 0.1]

alphas_num = [None] * len(deltas) * len(lambdas)
final_errors_mean = [None] * len(deltas) * len(lambdas)
final_errors_std = [None] * len(deltas) * len(lambdas)

for idx, l in enumerate(tqdm(lambdas, desc="lambda", leave=False)):
    for jdx, (delta_small, delta_large) in enumerate(
        tqdm(deltas, desc="delta", leave=False)
    ):
        i = idx * len(deltas) + jdx

        file_path = os.path.join(
            experiments_path,
            file_name_generator(
                loss_chosen,
                eps,
                alpha_min,
                alpha_max,
                alpha_points_num,
                d,
                reps,
                delta_small,
                delta_large,
                l,
                experiments=True,
            ),
        )

        file_exists = os.path.exists(file_path + ".npz")

        if not file_exists:
            (
                alphas_num[i],
                final_errors_mean[i],
                final_errors_std[i],
            ) = num.generate_different_alpha(
                num.find_coefficients_ridge,
                delta=delta_small,
                alpha_1=alpha_min,
                alpha_2=alpha_max,
                n_features=d,
                n_alpha_points=alpha_points_num,
                repetitions=reps,
                lambda_reg=l,
            )

            np.savez(
                file_path,
                alphas=alphas_num[i],
                error_mean=final_errors_mean[i],
                error_std=final_errors_std[i],
            )
        else:
            stored_data = np.load(file_path + ".npz")

            alphas_num[i] = stored_data["alphas"]
            final_errors_mean[i] = stored_data["error_mean"]
            final_errors_std[i] = stored_data["error_std"]

alphas_theory = [None] * len(deltas) * len(lambdas)
errors_theory = [None] * len(deltas) * len(lambdas)

for idx, l in enumerate(tqdm(lambdas, desc="lambda", leave=False)):
    for jdx, (delta_small, delta_large) in enumerate(
        tqdm(deltas, desc="delta", leave=False)
    ):
        i = idx * len(deltas) + jdx

        file_path = os.path.join(
            theory_path,
            file_name_generator(
                loss_chosen,
                eps,
                alpha_min,
                alpha_max,
                alpha_points_theory,
                d,
                reps,
                delta_small,
                delta_large,
                l,
                experiments=False,
            ),
        )

        file_exists = os.path.exists(file_path + ".npz")

        if not file_exists:
            while True:
                m = np.random.random()
                q = np.random.random()
                sigma = np.random.random()
                if np.square(m) < q + delta_small * q:
                    break

            initial = [m, q, sigma]

            (
                alphas_theory[i],
                errors_theory[i],
            ) = fpe.projection_ridge_different_alpha_theory(
                fpe.var_func_L2,
                fpe.var_hat_func_L2_num,
                alpha_1=alpha_min,
                alpha_2=alpha_max,
                n_alpha_points=alpha_points_theory,
                lambd=l,
                delta=delta_small,
                initial_cond=initial,
                verbose=True,
            )

            np.savez(file_path, alphas=alphas_theory[i], errors=errors_theory[i])
        else:
            stored_data = np.load(file_path + ".npz")

            alphas_theory[i] = stored_data["alphas"]
            errors_theory[i] = stored_data["errors"]

fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)

for idx, l in enumerate(lambdas):
    colormap = get_cmap(len(deltas) + 3, name=names_cm[idx])

    for jdx, delta in enumerate(deltas):
        i = idx * len(deltas) + jdx
        ax.plot(
            alphas_theory[i],
            errors_theory[i],
            label=r"$\lambda = {}$ $\Delta = {}$".format(l, delta),
            color=colormap(jdx + 3),
        )

        ax.errorbar(
            alphas_num[i],
            final_errors_mean[i],
            final_errors_std[i],
            marker=".",
            linestyle="None",
            color=colormap(jdx + 3),
        )

ax.set_title("L2 Loss")
ax.set_ylabel(r"$\frac{1}{d} E[||\hat{w} - w^\star||^2]$")
ax.set_xlabel(r"$\alpha$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([0.009, 110])
ax.minorticks_on()
ax.grid(True, which="both")
ax.legend()

fig.savefig("./imgs/togheter - {}.png".format(random_number), format="png")

plt.show()
