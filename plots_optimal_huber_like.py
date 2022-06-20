from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from src.utils import load_file
import src.plotting_utils as pu
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker


# def myfmt(x, pos):
#     return "{0:.1f}".format(x)
width = 1.0 * 458.63788
save = False
random_number = np.random.randint(100)

marker_cycler = ["*", "s", "P", "P", "v", "D"]

deltas_large = [10.0]  # 0.5, 1.0, 2.0, 5.0, 10.0
percentages = [0.01, 0.05, 0.1, 0.3]  # 0.01, 0.05, 0.1, 0.3

percentages.reverse()

all_params = list(product(deltas_large, percentages))

delta = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
# delta.reverse()

percentage, delta_small, delta_large = 0.1, 0.1, 5.0
eps, beta = percentage, 0.5
delta_large = [0.5, 1.0, 2.0, 5.0, 10.0]

experiments_settings = [
    {
        # "loss_name": "L2",
        "alpha_min": 0.01,
        "alpha_max": 100,
        "alpha_pts": 20,
        # "alpha_pts_theoretical": 36,
        # "alpha_pts_experimental": 4,
        # "reg_param": 1.0,
        # "delta": 0.5,
        "delta_small": delta_small,
        "delta_large": dl,
        "percentage": percentage,
        # "n_features": 500,
        #  "repetitions": 4,
        "beta": beta,
        "experiment_type": "BO",
    }
    for dl in delta_large
]

A = 1.0
indice = 3
alphas, _ = load_file(**experiments_settings[indice])
deltas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# alphas = np.flip(alphas) optimal_loss_huber_like_1.0_0.1_0.1_5.0_0.0.npz
files_s = [
    "dump/connected_optimal_loss_huber_like_{}_{}.npz".format(A, d) for d in deltas
]
files_a = [
    "dump/ass_connected_optimal_loss_huber_like_{}_{}.npz".format(A, d) for d in deltas
]


data_symm = [np.load(file_name) for file_name in files_s]
data_asymm = [np.load(file_name) for file_name in files_a]

N = 10
step = 0.1
# lw = 2.0
# delta = 0.1

pu.initialization_mpl()

fig1, ax1 = plt.subplots(
    1, 1, figsize=pu.set_size(width, fraction=0.49), tight_layout=True,
)
fig2, ax2 = plt.subplots(
    nrows=1, ncols=1, figsize=pu.set_size(width, fraction=0.49), tight_layout=True,
)

cmap = matplotlib.cm.get_cmap("plasma")  # len(files_s) + 1
cmap_disc = matplotlib.cm.get_cmap("plasma", len(files_s) + 1)
color_lines = []
error_names = []
error_names_latex = []

reg_param_lines = []

norm = mpl.colors.Normalize(vmin=0, vmax=10)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

for idx, (data_s, data_a) in enumerate(zip(data_symm, data_asymm)):
    ax1.plot(
        data_s["x"],
        data_s["loss"],
        linestyle="solid",
        color=cmap_disc(idx),
        label=r"$\alpha$ = {:03.2f}".format((alphas[idx])),
    )

    ax2.plot(
        data_a["x"],
        data_a["loss"],
        linestyle="solid",
        # marker="P",
        color=cmap_disc(idx),
        label=r"$\alpha$ = {:03.2f}".format((alphas[idx])),
    )

ax1.set_ylabel(r"$\ell_{\mathrm{opt}}(0,z)$")
ax1.set_xlabel(r"$z$")
# ax1.set_xscale("log")
# ax1.set_yscale("log")
ax1.set_xlim([-20, 20])
ax1.set_ylim([0, 10])
# ax1.legend(prop={"size": 22})

# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax2.set_ylabel(r"$\ell_{\mathrm{opt}}(0,z)$")
ax2.set_xlabel(r"$z$")
ax2.set_xscale("log")
ax2.set_yscale("log")
# ax2.legend(prop={"size": 22})

cbar = fig1.colorbar(
    sm,
    ticks=np.linspace(0, 10, 5)
    # ticks=np.logspace(-2, 2, 10),# (0.2, 2.1, step),  # np.logspace(-2, 2, 10),  #
    # boundaries=np.logspace(0.15, 2.15, step),
    # format=ticker.FuncFormatter(myfmt),
)

cbar.ax.set_title(r"$\Delta$")
cbar.ax.minorticks_off()

cbar2 = fig2.colorbar(
    sm,
    ticks=np.linspace(0, 10, 5)
    # ticks=np.logspace(-2, 2, 10),# (0.2, 2.1, step),  # np.logspace(-2, 2, 10),  #
    # boundaries=np.logspace(0.15, 2.15, step),
    # format=ticker.FuncFormatter(myfmt),
)

cbar2.ax.set_title(r"$\Delta$")
cbar2.ax.minorticks_off()

if save:
    pu.save_plot(
        fig1, "optimal_losses_huber_like_connected_different_A_{}".format(A),
    )
    # pu.save_plot(
    #     fig2, "loglog_optimal_huber_like_connected_different_A_{}".format(A),
    # )
plt.show()
