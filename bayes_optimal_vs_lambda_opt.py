import numpy as np
import matplotlib.pyplot as plt
from optimal_lambda import optimal_lambda
from tqdm.auto import tqdm
import fixed_point_equations_double as fixedpoint
from src.utils import check_saved, save_file, load_file

names_cm = ["Blues", "Reds", "Greens", "Oranges"]

random_number = np.random.randint(0, 100)

alpha_min, alpha_max = 0.01, 100
eps = [0.01, 0.1, 0.3]
alpha_points = 21
deltas = [
    [1.0, 3.0],
    [1.0, 4.0],
    [0.5, 2.0],
    [0.5, 3.0],
    [0.5, 2.5],
]
# [0.5, 1.5], [1.0, 2.0], [1.0, 5.0]

alphas_Hub = [None] * len(deltas) * len(eps)
errors_Hub = [None] * len(deltas) * len(eps)
lambdas_Hub = [None] * len(deltas) * len(eps)

colormap = fixedpoint.get_cmap(len(eps) * len(deltas) + 1)

# evaluates the lambda optimal for each value
print("-- Optimal reg_param Huber")
for idx, e in enumerate(tqdm(eps, desc="epsilon", leave=False)):
    for jdx, (delta_small, delta_large) in enumerate(
        tqdm(deltas, desc="delta", leave=False)
    ):
        i = idx * len(deltas) + jdx

        experiment_dict = {
            "loss_name": "Huber",
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "alpha_pts": alpha_points,
            "delta_small": delta_small,
            "delta_large": delta_large,
            "epsilon": e,
            "experiment_type": "reg param optimal",
        }

        file_exists, file_path = check_saved(**experiment_dict)

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

            alphas_Hub[i], errors_Hub[i], lambdas_Hub[i] = optimal_lambda(
                fixedpoint.var_func_L2,
                fixedpoint.var_hat_func_Huber_num_eps,
                alpha_1=alpha_min,
                alpha_2=alpha_max,
                n_alpha_points=alpha_points,
                delta_small=delta_small,
                delta_large=delta_large,
                initial_cond=initial,
                eps=e,
                verbose=True,
            )

            experiment_dict.update(
                {
                    "file_path": file_path,
                    "alphas": alphas_Hub[i],
                    "errors": errors_Hub[i],
                    "lambdas": lambdas_Hub[i],
                }
            )

            save_file(**experiment_dict)
        else:
            experiment_dict.update({"file_path": file_path})

            alphas_Hub[i], errors_Hub[i], lambdas_Hub[i] = load_file(**experiment_dict)


alphas_L2 = [None] * len(deltas) * len(eps)
errors_L2 = [None] * len(deltas) * len(eps)
lambdas_L2 = [None] * len(deltas) * len(eps)

colormap = fixedpoint.get_cmap(len(eps) * len(deltas) + 1)

# evaluates the lambda optimal for each value
print("-- Optimal reg_param L2")
for idx, e in enumerate(tqdm(eps, desc="epsilon", leave=False)):
    for jdx, (delta_small, delta_large) in enumerate(
        tqdm(deltas, desc="delta", leave=False)
    ):
        i = idx * len(deltas) + jdx

        experiment_dict = {
            "loss_name": "L2",
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "alpha_pts": alpha_points,
            "delta_small": delta_small,
            "delta_large": delta_large,
            "epsilon": e,
            "experiment_type": "reg param optimal",
        }

        file_exists, file_path = check_saved(**experiment_dict)

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

            alphas_L2[i], errors_L2[i], lambdas_L2[i] = optimal_lambda(
                fixedpoint.var_func_L2,
                fixedpoint.var_hat_func_L2_num_eps,
                alpha_1=alpha_min,
                alpha_2=alpha_max,
                n_alpha_points=alpha_points,
                delta_small=delta_small,
                delta_large=delta_large,
                initial_cond=initial,
                eps=e,
                verbose=True,
            )

            experiment_dict.update(
                {
                    "file_path": file_path,
                    "alphas": alphas_L2[i],
                    "errors": errors_L2[i],
                    "lambdas": lambdas_L2[i],
                }
            )

            save_file(**experiment_dict)
        else:
            experiment_dict.update({"file_path": file_path})

            alphas_L2[i], errors_L2[i], lambdas_L2[i] = load_file(**experiment_dict)

# evaluates the bayes optimal
alphas_BO = [None] * len(deltas) * len(eps)
errors_BO = [None] * len(deltas) * len(eps)

print("-- BO")
for idx, e in enumerate(tqdm(eps, desc="epsilon", leave=False)):
    for jdx, (delta_small, delta_large) in enumerate(
        tqdm(deltas, desc="delta", leave=False)
    ):
        i = idx * len(deltas) + jdx

        experiment_dict = {
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "alpha_pts": alpha_points,
            "delta_small": delta_small,
            "delta_large": delta_large,
            "epsilon": e,
            "experiment_type": "BO",
        }

        file_exists, file_path = check_saved(**experiment_dict)

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
                alphas_BO[i],
                errors_BO[i],
            ) = fixedpoint.projection_ridge_different_alpha_theory(
                fixedpoint.var_func_BO,
                fixedpoint.var_hat_func_BO_num_eps,
                alpha_1=alpha_min,
                alpha_2=alpha_max,
                n_alpha_points=alpha_points,
                lambd=1.0,
                delta_small=delta_small,
                delta_large=delta_large,
                initial_cond=initial,
                verbose=True,
                eps=e,
            )

            experiment_dict.update(
                {"file_path": file_path, "alphas": alphas_BO[i], "errors": errors_BO[i],}
            )

            save_file(**experiment_dict)
        else:
            experiment_dict.update({"file_path": file_path})

            alphas_BO[i], errors_BO[i] = load_file(**experiment_dict)

# colormap_BO = fixedpoint.get_cmap(len(deltas) + 2, name="Greys")

# colormap = fixedpoint.get_cmap(len(deltas) + 2, name="Reds")

# colormap_L2 = fixedpoint.get_cmap(len(deltas) + 2, name="Greens")

# for idx, e in enumerate(eps):

#     for jdx, delta in enumerate(deltas):
#         fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)

#         i = idx * len(deltas) + jdx

#         ax.plot(
#             alphas[i],
#             errors[i],
#             # marker=".",
#             label=r"Huber",
#             color="g",  #  colormap(jdx + 2),
#         )

#         ax.plot(
#             alphas_L2[i],
#             errors_L2[i],
#             # marker=".",
#             label=r"L2",
#             color="r",  #  colormap_L2(jdx + 2),
#         )

#         ax.plot(
#             alphasBO[i],
#             errorsBO[i],
#             #  marker=".",
#             linestyle="dashed",
#             label=r"BO",
#             color="k",  #  colormap_BO(jdx + 2),
#         )

#         ax.set_title(
#             r"L2 vs. Huber $\lambda_{{opt}}$ vs. BO $\epsilon$ = {} $\Delta$ = {}".format(
#                 e, delta
#             )
#         )
#         ax.set_ylabel(r"$\frac{1}{d} E[||\hat{w} - w^\star||^2]$")
#         ax.set_xlabel(r"$\alpha$")
#         ax.set_xscale("log")
#         ax.set_yscale("log")
#         ax.minorticks_on()
#         ax.grid(True, which="both")
#         ax.legend()

#         fig.savefig(
#             "./imgs/confront_epsil_{}_delta_{}_{}.png".format(
#                 e, delta, random_number
#             ),
#             format="png",
#         )

#         plt.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)
figL2BO, axL2BO = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)
figHubBO, axHubBO = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)

colormap_BO = fixedpoint.get_cmap(len(eps) * len(deltas) + 2, name="Greys")

colormap = fixedpoint.get_cmap(len(eps) * len(deltas) + 2, name="Reds")

colormap_L2 = fixedpoint.get_cmap(len(eps) * len(deltas) + 2, name="Greens")

for idx, e in enumerate(eps):
    for jdx, delta in enumerate(deltas):
        i = idx * len(deltas) + jdx

        ax.plot(
            alphas_Hub[i],
            errors_Hub[i],
            # marker=".",
            label=r"Huber $\epsilon = {}$ $\Delta = {}$".format(e, delta),
            color=colormap(i + 2),
            linewidth=2,
        )

for idx, e in enumerate(eps):
    for jdx, delta in enumerate(deltas):
        i = idx * len(deltas) + jdx

        ax.plot(
            alphas_L2[i],
            errors_L2[i],
            # marker=".",
            label=r"L2 $\epsilon = {}$ $\Delta = {}$".format(e, delta),
            color=colormap_L2(i + 2),
            linewidth=2,
        )

for idx, e in enumerate(eps):
    for jdx, delta in enumerate(deltas):
        i = idx * len(deltas) + jdx
        ax.plot(
            alphas_BO[i],
            errors_BO[i],
            #  marker=".",
            linestyle="dashed",
            label=r"BO $\epsilon = {}$ $\Delta = {}$".format(e, delta),
            color=colormap_BO(i + 2),
        )

ax.set_title(r"L2 vs. Huber $\lambda_{opt}$ vs. BO")
ax.set_ylabel(r"$\frac{1}{d} E[||\hat{w} - w^\star||^2]$")
ax.set_xlabel(r"$\alpha$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.minorticks_on()
ax.grid(True, which="both")
ax.legend()

# fig.savefig("./imgs/HubL2lambdaopt_{}.png".format(random_number), format="png")

for idx, e in enumerate(eps):
    cmap = fixedpoint.get_cmap(len(deltas) + 1, name=names_cm[idx])
    for jdx, delta in enumerate(deltas):
        i = idx * len(deltas) + jdx
        axL2BO.plot(
            alphas_L2[i],
            (errors_L2[i] - errors_BO[i]) / errors_BO[i],
            linestyle="solid",
            marker=".",
            color=cmap(jdx + 1),
            label=r"$\epsilon = {}$ $\Delta = {}$".format(e, delta),
        )

axL2BO.set_title(r"L2 $\lambda_{opt}$ vs. BO ")
axL2BO.set_ylabel(
    r"$(\frac{1}{d} E[||\hat{w} - w^\star||^2]_{L2} - \frac{1}{d} E[||\hat{w} - w^\star||^2]_{BO} ) / E[||\hat{w} - w^\star||^2]_{BO}$"
)
axL2BO.set_xlabel(r"$\alpha$")
axL2BO.set_xscale("log")
axL2BO.set_yscale("log")
axL2BO.minorticks_on()
axL2BO.grid(True, which="both")
axL2BO.legend()

# figL2BO.savefig("./imgs/L2vslambdaopt_log_{}.png".format(random_number), format="png")

for idx, e in enumerate(eps):
    cmap = fixedpoint.get_cmap(len(deltas) + 1, name=names_cm[idx])
    for jdx, delta in enumerate(deltas):
        i = idx * len(deltas) + jdx
        axHubBO.plot(
            alphas_Hub[i],
            (errors_Hub[i] - errors_BO[i]) / errors_BO[i],
            marker=".",
            color=cmap(jdx + 1),
            label=r"$\epsilon = {}$ $\Delta = {}$".format(e, delta),
        )

axHubBO.set_title(r"Huber $\lambda_{opt}$ vs. BO")
axHubBO.set_ylabel(
    r"$(E[||\hat{w} - w^\star||^2]_{Huber} - E[||\hat{w} - w^\star||^2]_{BO} ) / E[||\hat{w} - w^\star||^2]_{BO}$"
)
axHubBO.set_xlabel(r"$\alpha$")
axHubBO.set_xscale("log")
axHubBO.set_yscale("log")
axHubBO.minorticks_on()
axHubBO.grid(True, which="both")
axHubBO.legend()

# figHubBO.savefig(
#     "./imgs/Hubvslambdaopt_log_{}.png".format(random_number), format="png"
# )

plt.show()
