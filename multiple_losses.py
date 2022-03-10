import numpy as np
import matplotlib.pyplot as plt
import fixed_point_equations_double as fpedbl
import numerics as num
from tqdm.auto import tqdm
import fixed_point_equations_bayes_opt as bofpe
import os
from src.utils import file_name_generator

random_number = np.random.randint(0, 100)

theory_path = "./data/theory"
experiments_path = "./data/experiments"
bo_path = "./data/bayes optimal"

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

names_cm = ["YlGnBu", "RdPu", "Greens", "Oranges", "Reds", "Purples", "Blues"]


def get_cmap(n, name="hsv"):
    return plt.cm.get_cmap(name, n)


loss_chosen = "L2"
alpha_min, alpha_max = 0.01, 100
epsil = 0.3
alpha_points_theory = 31
d = 500
reps = 20
deltas = [[0.5, 1.5]]  # , [0.5, 2.5], [1.0, 2.0]
lambdas = [0.5, 1.0, 1.5, 2.0]

alphas_theory = [None] * len(deltas) * len(lambdas)
errors_theory = [None] * len(deltas) * len(lambdas)

print("--- theory {}".format(loss_chosen))

for idx, l in enumerate(tqdm(lambdas, desc="lambda", leave=False)):
    for jdx, [delta_small, delta_large] in enumerate(
        tqdm(deltas, desc="delta", leave=False)
    ):
        i = idx * len(deltas) + jdx

        file_path = os.path.join(
            theory_path,
            file_name_generator(
                loss_chosen,
                alpha_min,
                alpha_max,
                alpha_points_theory,
                d,
                reps,
                l,
                delta_small,
                delta_large=delta_large,
                eps=epsil,
                experiment_type="theory",
            ),
        )

        file_exists = os.path.exists(file_path + ".npz")

        if not file_exists:
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

            (
                alphas_theory[i],
                errors_theory[i],
            ) = fpedbl.projection_ridge_different_alpha_theory(
                fpedbl.var_func_L2,
                fpedbl.var_hat_func_L2_num_eps,
                alpha_1=alpha_min,
                alpha_2=alpha_max,
                n_alpha_points=alpha_points_theory,
                lambd=l,
                delta_small=delta_small,
                delta_large=delta_large,
                initial_cond=initial,
                eps=epsil,
                verbose=True,
            )

            np.savez(file_path, alphas=alphas_theory[i], errors=errors_theory[i])
        else:
            stored_data = np.load(file_path + ".npz")

            alphas_theory[i] = stored_data["alphas"]
            errors_theory[i] = stored_data["errors"]

alphas_BO = [None] * len(deltas)
errors_BO = [None] * len(deltas)

print("--- BO")

for jdx, [delta_small, delta_large] in enumerate(tqdm(deltas, desc="delta", leave=False)):
    file_path = os.path.join(
        bo_path,
        file_name_generator(
            "BO",
            alpha_min,
            alpha_max,
            alpha_points_theory,
            d,
            reps,
            l,
            delta_small,
            delta_large=delta_large,
            eps=epsil,
            experiment_type="BO",
        ),
    )

    file_exists = os.path.exists(file_path + ".npz")

    if not file_exists:
        while True:
            m = 0.89 * np.random.random() + 0.1
            q = 0.89 * np.random.random() + 0.1
            sigma = 0.89 * np.random.random() + 0.1
            if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
                break

        initial = [m, q, sigma]

        alphas_BO[jdx], errors_BO[jdx] = fpedbl.projection_ridge_different_alpha_theory(
            fpedbl.var_func_BO,
            fpedbl.var_hat_func_BO_num_eps,
            alpha_1=alpha_min,
            alpha_2=alpha_max,
            n_alpha_points=alpha_points_theory,
            lambd=l,
            delta_small=delta_small,
            delta_large=delta_large,
            initial_cond=initial,
            eps=epsil,
            verbose=True,
        )

        # alphas_BO[jdx], errors_BO[jdx] = bofpe.bayes_opt_theory(
        #     bofpe.var_hat_func_bayes_opt_eps,
        #     alpha_1=alpha_min,
        #     alpha_2=alpha_max,
        #     n_alpha_points=alpha_points_theory,
        #     delta_small=delta_small,
        #     delta_large=delta_large,
        #     initial_cond=initial,
        #     verbose=True,
        # )

        np.savez(file_path, alphas=alphas_theory[i], errors=errors_theory[i])
    else:
        stored_data = np.load(file_path + ".npz")

        alphas_BO[jdx] = stored_data["alphas"]
        errors_BO[jdx] = stored_data["errors"]


fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)

for idx, l in enumerate(lambdas):
    colormap = get_cmap(len(deltas) + 3, name=names_cm[idx])

    for jdx, delta in enumerate(deltas):
        i = idx * len(deltas) + jdx
        ax.plot(
            alphas_theory[i],
            errors_theory[i],
            # marker='.',
            label=r"$\lambda = {}$ $\Delta = {}$".format(l, delta),
            color=colormap(jdx + 3),
        )

colormapBO = get_cmap(len(deltas) + 3, "Greys")

for jdx, delta in enumerate(deltas):
    ax.plot(
        alphas_BO[jdx],
        errors_BO[jdx],
        # marker='.',
        label=r"BO $\Delta$ = {}".format(delta),
        color=colormapBO(jdx + 3),
    )

ax.set_title(r"L2 Loss - $\epsilon = {:.2f}$".format(epsil))
ax.set_ylabel(r"$\frac{1}{d} E[||\hat{w} - w^\star||^2]$")
ax.set_xlabel(r"$\alpha$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([0.009, 110])
ax.minorticks_on()
ax.grid(True, which="both")
ax.legend()

fig.savefig(
    "./imgs/together - double noise - code {}.png".format(random_number),
    format="png",
    dpi=150,
)

plt.show()
