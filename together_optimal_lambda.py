# from matplotlib import markers
from matplotlib.lines import Line2D
from src.utils import load_file
import src.plotting_utils as pu
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

save = False
random_number = np.random.randint(100)

marker_cycler = ['*', 's', "P", "P", "v", "D"]

delta = 1.0
deltas_large = [0.5, 1.0, 2.0, 5.0, 10.0]
percentages = [0.3] # 0.01, 0.05, 0.1,

aa = [0.5, 1.0, 1.5]

all_params = list(product(deltas_large, percentages))
# all_params = [(5.0, 0.3)] # [(0.5, 0.01), (0.5, 0.05), (1.0, 0.01), (2.0, 0.01)] # (0.5, 0.01), (0.5, 0.05), (0.5, 0.1), (1.0, 0.01), 
# (0.5, 0.01), (0.5, 0.05), (0.5, 0.1), (0.5, 0.3), (1.0, 0.01), (1.0, 0.05), (1.0, 0.1), (1.0, 0.3), (2.0, 0.01), (2.0, 0.05), (2.0, 0.1), (2.0, 0.3), (5.0, 0.01), (5.0, 0.05), (5.0, 0.1), (5.0, 0.3), (10.0, 0.01), (10.0, 0.05), (10.0, 0.1)

L2_settings = [
    {
        "loss_name": "L2",
        "alpha_min": 0.01,
        "alpha_max": 100,
        "alpha_pts": 36,
        "reg_param" : 1.0,
        "delta": 0.5,
        # "delta_small": 0.1,
        # "delta_large": dl,
        # "percentage": p,
        "experiment_type": "theory",
    }
]

Huber_settings = [
    {
        "loss_name": "Huber",
        "alpha_min": 0.01,
        "alpha_max": 100,
        "alpha_pts": 36,
        "delta" : 0.5,
        "reg_param" : 1.0,
        "a" : a,
        # "delta_small": 0.1,
        # "delta_large": dl,
        # "percentage": p,
        "experiment_type": "theory",
    }
    for a in aa # for dl, p in product(deltas_large, percentages)
]

BO_settings = [
    {
        "alpha_min": 0.01,
        "alpha_max": 100,
        "alpha_pts": 36,
        "delta" : 0.5,
        # "delta_small": 0.1,
        # "delta_large": dl,
        # "percentage": p,
        "experiment_type": "BO",
    }
]

pu.initialization_mpl()

fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True,)
fig2, (ax20, ax21) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(9, 8), gridspec_kw={'height_ratios': [1,1]})
# fig2, ax21 = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True,)
# ax22 = ax21.twinx()

cmap = plt.get_cmap("tab10")
color_lines = []
error_names = []

reg_param_lines = []

alphas_L2, errors_L2 = load_file(**L2_settings[0])
alphas_BO, errors_BO = load_file(**BO_settings[0])

ax.plot(
    alphas_L2, 
    errors_L2, 
    linestyle='dotted',
    # color=cmap(idx),
)

ax.plot(
    alphas_BO, 
    errors_BO,
    linestyle='solid',
    # color=cmap(idx),
)

for idx, (Huber_d,) in enumerate(zip(Huber_settings)):
    alphas_Huber, errors_Huber = load_file(**Huber_d)

    color_lines.append(Line2D([0], [0], color=cmap(idx)))
    error_names.append("$\Delta_\ell = {:.1f}$".format(all_params[idx][0]))

    ax.plot(
        alphas_Huber,
        errors_Huber,
        linestyle='dashed',
        # color=cmap(idx),
        label="a = {:.1f}".format(aa[idx])
    )

   

    ax20.plot(
        alphas_L2, 
        errors_L2 - errors_BO, 
        linestyle='dotted',
        marker=marker_cycler[idx],
        linewidth=1.0,
        color='b',
        label="$\Delta_\ell = {:.1f} \:\epsilon = {:.2f}$ (L2)".format(*all_params[idx])
    )

    ax20.plot(
        alphas_L2, 
        errors_Huber - errors_BO, 
        linestyle='dotted',
        marker=marker_cycler[idx],
        linewidth=1.0,
        color='b',
        label="$\Delta_\ell = {:.1f} \:\epsilon = {:.2f}$ (L2)".format(*all_params[idx])
    )

    ax20.plot(
        alphas_L2, 
        errors_Huber - errors_BO, 
        linestyle='dotted',
        marker=marker_cycler[idx],
        linewidth=1.0,
        color='b',
        label="$\Delta_\ell = {:.1f} \:\epsilon = {:.2f}$ (L2)".format(*all_params[idx])
    )

# loss_lines = [
#     Line2D([0], [0], color='k', linestyle='dotted'),
#     Line2D([0], [0], color='k', linestyle='dashed'),
#     Line2D([0], [0], color='k', linestyle='solid')
# ]

# loss_lines_reduced = [
#     Line2D([0], [0], color='k', linestyle='dotted'),
#     Line2D([0], [0], color='k', linestyle='dashed'),
# ]

# first_legend = ax.legend(loss_lines, ["L2", "Huber", "BayesOptimal"])
# ax.add_artist(first_legend)
# ax.legend(color_lines, error_names, loc="lower left")

ax.set_ylabel(r"Generalization Error: $\frac{1}{d} \mathbb{E}\qty[\norm{\bf{\hat{w}} - \bf{w^\star}}^2]$")
ax.set_xlabel(r"$\alpha$")
ax.set_xscale("log")
ax.set_yscale("log")
# ax.set_xlim([0.009, 110])
ax.legend()

fig2.subplots_adjust(wspace=0, hspace=0)

ax20.set_ylabel("Optimal Regularization Parameter", color='b')
ax21.set_ylabel("Optimal Huber Prameter", color='g')
ax20.set_xlabel(r"$\alpha$")
ax20.set_xscale("log")
# ax21.set_yscale("log")
# ax21.set_xlim([0.009, 110])
ax20.legend()

if save:
    pu.save_plot(fig, "optimal_confronts")
    pu.save_plot(fig2, "optimal_confronts_params")

plt.show()
