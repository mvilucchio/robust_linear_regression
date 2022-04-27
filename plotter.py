# from matplotlib import markers
from src.utils import load_file
import src.plotting_utils as pu
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

save = False
random_number = np.random.randint(100)

deltas_large = [0.5, 1.0, 2.0, 5.0, 10.0]
percentages = [0.01, 0.05, 0.1, 0.3]

all_params = list(product(deltas_large, percentages))
# all_params = [(5.0, 0.3)] # [(0.5, 0.01), (0.5, 0.05), (1.0, 0.01), (2.0, 0.01)] # (0.5, 0.01), (0.5, 0.05), (0.5, 0.1), (1.0, 0.01), 
# (0.5, 0.01), (0.5, 0.05), (0.5, 0.1), (0.5, 0.3), (1.0, 0.01), (1.0, 0.05), (1.0, 0.1), (1.0, 0.3), (2.0, 0.01), (2.0, 0.05), (2.0, 0.1), (2.0, 0.3), (5.0, 0.01), (5.0, 0.05), (5.0, 0.1), (5.0, 0.3), (10.0, 0.01), (10.0, 0.05), (10.0, 0.1)

# L2_settings = [
#     {
#         "loss_name": "L2",
#         "alpha_min": 0.01,
#         "alpha_max": 100,
#         "alpha_pts": 36,
#         # "delta": 0.1,
#         "delta_small": 0.1,
#         "delta_large": dl,
#         "percentage": p,
#         "experiment_type": "reg_param optimal",
#     }
#     for dl, p in all_params # product(deltas_large, percentages)
# ]

# Huber_settings = [
#     {
#         "loss_name": "Huber",
#         "alpha_min": 0.01,
#         "alpha_max": 100,
#         "alpha_pts": 12,
#         "delta_small": 0.1,
#         "delta_large": dl,
#         "percentage": p,
#         "experiment_type": "reg_param huber_param optimal",
#     }
#     for dl, p in all_params
# ]

BO_settings = [
    {
        "alpha_min": 0.01,
        "alpha_max": 100,
        "alpha_pts": 36,
        "delta_small": 0.1,
        "delta_large": dl,
        "percentage": p,
        "experiment_type": "BO",
    }
    for dl, p in product(deltas_large, percentages)
]

pu.initialization_mpl()

fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True,)
fig2, ax21 = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True,)
ax22 = ax21.twinx()

cmap = plt.get_cmap("tab10")

for idx, (BO_d, ) in enumerate(zip(BO_settings)): # L2_d, , BO_d   , , BO_settings
    # alphas_L2, errors_L2, lambdas_L2 = load_file(**L2_d)
    # alphas_Huber, errors_Huber, lambdas_Huber, huber_params = load_file(**Huber_d)
    alphas_BO, errors_BO = load_file(**BO_d)

    # ax.plot(
    #     alphas_L2, 
    #     errors_L2, 
    #     marker=".",
    #     # linestyle='dotted',
    #     color=cmap(idx),
    #     label="$\Delta_\ell = {:.1f} \:\epsilon = {:.2f}$".format(*all_params[idx])
    # )

    # ax.plot(
    #     alphas_Huber,
    #     errors_Huber,
    #     linestyle='dashed',
    #     color=cmap(idx),
    #     label="$\Delta_\ell = {:.1f} \:\epsilon = {:.2f}$".format(*all_params[idx])
    # )

    ax.plot(
        alphas_BO, 
        errors_BO,
        linestyle='solid',
        color=cmap(idx),
        label="$\Delta_\ell = {:.1f} \:\epsilon = {:.2f}$".format(*all_params[idx])
    )

    # ax21.plot(
    #     alphas_L2, 
    #     lambdas_L2, 
    #     linestyle='dotted',
    #     color='b',
    #     label="$\Delta_\ell = {:.1f} \:\epsilon = {:.2f}$".format(*all_params[idx])
    # )

    # ax21.plot(
    #     alphas_Huber,
    #     lambdas_Huber,
    #     linestyle='dashed',
    #     color='b',
    #     label="$\Delta_\ell = {:.1f} \:\epsilon = {:.2f}$".format(*all_params[idx])
    # )
    
    # ax22.plot(
    #     alphas_Huber,
    #     huber_params,
    #     linestyle='dashed',
    #     color='g',
    #     label="$\Delta_\ell = {:.1f} \:\epsilon = {:.2f}$".format(*all_params[idx])
    # )

# huber_line = Line2D([0], [0], c='r', lw=1)


ax.set_ylabel(r"Generalization Error: $\frac{1}{d} \mathbb{E}\qty[\norm{\bf{\hat{w}} - \bf{w^\star}}^2]$")
ax.set_xlabel(r"$\alpha$")
ax.set_xscale("log")
ax.set_yscale("log")
# ax.set_xlim([0.009, 110])
ax.legend()

ax21.set_ylabel("Optimal Regularization Parameter", color='b')
ax22.set_ylabel("Optimal Huber Prameter", color='g')
ax21.set_xlabel(r"$\alpha$")
ax21.set_xscale("log")
# ax21.set_yscale("log")
# ax21.set_xlim([0.009, 110])
ax21.legend()

if save:
    pu.save_plot(fig, "optimal_confronts")
    pu.save_plot(fig2, "optimal_confronts_params")

plt.show()
