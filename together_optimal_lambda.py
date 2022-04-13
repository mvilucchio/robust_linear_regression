import numpy as np
import matplotlib.pyplot as plt
import fixed_point_equations_double as fpedbl
from optimal_lambda import optimal_lambda
import numerics as num
from tqdm.auto import tqdm
from src.utils import check_saved, load_file, save_file

random_number = np.random.randint(0, 100)

names_cm = ["Purples", "Blues", "Greens", "Oranges", "Greys"]


def get_cmap(n, name="hsv"):
    return plt.cm.get_cmap(name, n)


alpha_points_theory = 36
d = 1500
reps = 40

random_number = np.random.randint(0, 100)

alpha_min, alpha_max = 0.01, 100
eps = [0.05, 0.3]
loss_chosen = "Huber"
deltas = [[0.1, 2.0], [0.1, 10.0]]

alphas_Hub = [None] * len(deltas) * len(eps)
errors_Hub = [None] * len(deltas) * len(eps)
lambdas_Hub = [None] * len(deltas) * len(eps)

colormap = fpedbl.get_cmap(len(eps) * len(deltas) + 1)

# load huber

# evaluates the lambda optimal for each value
for idx, e in enumerate(tqdm(eps, desc="epsilon", leave=False)):
    for jdx, (delta_small, delta_large) in enumerate(
        tqdm(deltas, desc="delta", leave=False)
    ):
        i = idx * len(deltas) + jdx

        dat = np.load(
            f"./data_plot/reg_param optimal/{loss_chosen} double noise - eps {e} - reg_param optimal - alphas [0.01 100 36] - delta [{delta_small} {delta_large}].npz"
        )

        alphas_Hub[i] = dat["alphas"]
        errors_Hub[i] = dat["errors"]
        lambdas_Hub[i] = dat["lambdas"]


alphas_num = [None] * len(deltas) * len(eps)
lambdas_num = [None] * len(deltas) * len(eps)
final_errors_mean = [None] * len(deltas) * len(eps)
final_errors_std = [None] * len(deltas) * len(eps)

for idx, e in enumerate(tqdm(eps, desc="lambda", leave=False)):
    for jdx, [delta_small, delta_large] in enumerate(
        tqdm(deltas, desc="delta", leave=False)
    ):
        i = idx * len(deltas) + jdx

        alphas_num[i] = alphas_Hub[i][::4].copy()
        lambdas_num[i] = lambdas_Hub[i][::4].copy()
        final_errors_mean[i] = np.empty_like(alphas_Hub[i][::4])
        final_errors_std[i] = np.empty_like(alphas_Hub[i][::4])

        n_alpha_points = len(alphas_num[i])

        final_errors_mean[i] = np.empty((n_alpha_points,))
        final_errors_std[i] = np.empty((n_alpha_points,))

        for udx, (alpha, reg_param) in enumerate(
            zip(tqdm(alphas_num[i], desc="alpha", leave=False), lambdas_num[i])
        ):
            all_gen_errors = np.empty((reps,))

            for kdx in tqdm(range(reps), desc="reps", leave=False):
                xs, ys, _, _, ground_truth_theta = num.data_generation_double_noise(
                    n_features=d,
                    n_samples=max(int(np.around(d * alpha)), 1),
                    n_generalization=1,
                    delta_small=delta_small,
                    delta_large=delta_large,
                    eps=e,
                )

                estimated_theta = num.find_coefficients_Huber(ys, xs, reg_param=reg_param)

                all_gen_errors[kdx] = np.divide(
                    np.sum(np.square(ground_truth_theta - estimated_theta)), d
                )

                del xs
                del ys
                del ground_truth_theta

            final_errors_mean[i][udx] = np.mean(all_gen_errors)
            final_errors_std[i][udx] = np.std(all_gen_errors)

            del all_gen_errors

        np.savez(
            ".experiments2-Huber-optimal-eps-{}-deltas-{}-deltal-{}".format(
                e, delta_small, delta_large
            ),
            alphas=alphas_num[i],
            errors_mean=final_errors_mean[i],
            errors_std=final_errors_std[i],
            lambdas=lambdas_num[i],
        )


fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)

for idx, e in enumerate(eps):
    colormap = get_cmap(len(deltas) + 3, name=names_cm[idx])

    for jdx, delta in enumerate(deltas):
        i = idx * len(deltas) + jdx
        ax.plot(
            alphas_Hub[i],
            errors_Hub[i],
            # marker='.',
            label=r"$\epsilon = {}$ $\Delta = {}$".format(e, delta),
            color=colormap(jdx + 3),
        )

        ax.errorbar(
            alphas_num[i],
            final_errors_mean[i],
            final_errors_std[i],
            marker=".",
            linestyle="None",
            # label=r"$\lambda = {}$ $\Delta = {}$".format(l, delta),
            color=colormap(jdx + 3),
        )

ax.set_title(r"{} Loss".format(loss_chosen))
ax.set_ylabel(r"$\frac{1}{d} E[||\hat{w} - w^\star||^2]$")
ax.set_xlabel(r"$\alpha$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([0.009, 110])
ax.minorticks_on()
ax.grid(True, which="both")
ax.legend()

fig.savefig(
    "./imgs/zzz{} - together - double noise - {}.png".format(loss_chosen, random_number),
    format="png",
    dpi=150,
)

plt.show()
