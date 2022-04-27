from matplotlib import markers
from src.utils import load_file
import matplotlib.pyplot as plt
import src.plotting_utils as pu
import numpy as np

save = False
lw = 2
random_number = np.random.randint(100)

deltas = [0.02, 0.05, 0.1, 0.12, 0.15] #, 0.15, 0.2

L2_settings = [
    {
        "loss_name": "L2",
        "alpha_min": 0.01,
        "alpha_max": 100,
        "alpha_pts": 200,
        "delta": d,
        # "delta_small": 0.1,
        # "delta_large": dl,
        # "percentage": p,
        "experiment_type": "reg_param optimal",
    }
    for d in deltas
]

pu.initialization_mpl()

fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(9, 8), gridspec_kw={'height_ratios': [3,1]})

# fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True,)
# fig2, ax21 = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True,)

cmap = plt.get_cmap("tab10")

for idx, (L2_d,) in enumerate(zip(L2_settings)): # L2_d, , BO_d   , , BO_settings
    alphas_L2, errors_L2, lambdas_L2 = load_file(**L2_d)

    differences = np.abs(lambdas_L2[1:] - lambdas_L2[:-1])
    jumps = differences >= 0.01
    indices = np.argwhere(jumps)

    ax0.plot(
        alphas_L2, 
        errors_L2, 
        linewidth=lw,
        marker=".",
        # linestyle='dotted',
        color=cmap(idx),
        label="$\Delta = {:.2f}$".format(deltas[idx])
    )

    for jdx in indices:
        ax0.axvline(
            x=(alphas_L2[jdx] + alphas_L2[jdx + 1]) / 2,
            color=cmap(idx),
            linestyle='dashed',
        )

    ax1.plot(
        alphas_L2, 
        lambdas_L2, 
        linewidth=lw,
        marker=".",
        # linestyle='dotted',
        # color='b',
        label="$\Delta = {:.2f}$".format(deltas[idx])
    )

fig.subplots_adjust(wspace=0, hspace=0)

ax0.set_ylabel(r"Generalization Error: $\frac{1}{d} \mathbb{E}\qty[\norm{\bf{\hat{w}} - \bf{w^\star}}^2]$")
# ax0.set_xlabel(r"$\alpha$")
ax0.set_xscale("log")
ax0.set_yscale("log")
# ax0.set_xlim([0.009, 110])
ax0.legend()

ax1.set_ylabel(r"$\lambda_{\text{opt}}$") #"Optimal Regularization Parameter"
ax1.set_xlabel(r"$\alpha$")
ax1.set_xscale("log")
# ax1.set_yscale("log")
# ax1.set_xlim([0.009, 110])
# ax1.legend()

if save:
    pu.save_plot(fig, "optimal_confronts_{:d}".format(random_number))


plt.show()
