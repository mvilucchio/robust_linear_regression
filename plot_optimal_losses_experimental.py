from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from src.utils import load_file
import src.plotting_utils as pu
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from itertools import product
import matplotlib as mpl

save = False
width = 1.0 * 458.63788
random_number = np.random.randint(100)


marker_cycler = ["*", "s", "P", "P", "v", "D"]

deltas_large = [10.0]  # 0.5, 1.0, 2.0, 5.0, 10.0
percentages = [0.01, 0.05, 0.1, 0.3]  # 0.01, 0.05, 0.1, 0.3

percentages.reverse()

all_params = list(product(deltas_large, percentages))
# all_params = [(5.0, 0.3)] # [(0.5, 0.01), (0.5, 0.05), (1.0, 0.01), (2.0, 0.01)] # (0.5, 0.01), (0.5, 0.05), (0.5, 0.1), (1.0, 0.01),
# (0.5, 0.01), (0.5, 0.05), (0.5, 0.1), (0.5, 0.3), (1.0, 0.01), (1.0, 0.05), (1.0, 0.1), (1.0, 0.3), (2.0, 0.01), (2.0, 0.05), (2.0, 0.1), (2.0, 0.3), (5.0, 0.01), (5.0, 0.05), (5.0, 0.1), (5.0, 0.3), (10.0, 0.01), (10.0, 0.05), (10.0, 0.1)

files_s = [
    "/Users/matteovilucchio/beta_0.csv",
    # "/Users/matteovilucchio/beta_005.csv",
    "/Users/matteovilucchio/beta_01.csv",
    "/Users/matteovilucchio/beta_02.csv",
    "/Users/matteovilucchio/beta_03.csv",
    "/Users/matteovilucchio/beta_04.csv",
    "/Users/matteovilucchio/beta_05.csv",
    "/Users/matteovilucchio/beta_06.csv",
    "/Users/matteovilucchio/beta_07.csv",
    "/Users/matteovilucchio/beta_08.csv",
    "/Users/matteovilucchio/beta_09.csv",
    "/Users/matteovilucchio/beta_1.csv",
]
files_a = [
    "/Users/matteovilucchio/log_beta_0.csv",
    # "/Users/matteovilucchio/log_beta_005.csv",
    "/Users/matteovilucchio/log_beta_01.csv",
    "/Users/matteovilucchio/log_beta_02.csv",
    "/Users/matteovilucchio/log_beta_03.csv",
    "/Users/matteovilucchio/log_beta_04.csv",
    "/Users/matteovilucchio/log_beta_05.csv",
    "/Users/matteovilucchio/log_beta_06.csv",
    "/Users/matteovilucchio/log_beta_07.csv",
    "/Users/matteovilucchio/log_beta_08.csv",
    "/Users/matteovilucchio/log_beta_09.csv",
    "/Users/matteovilucchio/log_beta_1.csv",
]

beta = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# beta.reverse()
# files_s.reverse()
# files_a.reverse()

data_symm = [np.genfromtxt(file_name, delimiter=",") for file_name in files_s]
data_asymm = [np.genfromtxt(file_name, delimiter=",") for file_name in files_a]

cmap = matplotlib.cm.get_cmap("plasma")  #
cmap_disc = matplotlib.cm.get_cmap("plasma", len(files_s) + 1)
print(len(files_s))
color_lines = []
error_names = []
error_names_latex = []

norm = mpl.colors.Normalize(vmin=0.0, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  #
sm.set_array([])

reg_param_lines = []

pu.initialization_mpl()

fig1, ax1 = plt.subplots(
    1, 1, figsize=pu.set_size(width, fraction=0.49), tight_layout=True,
)
fig2, ax2 = plt.subplots(
    nrows=1, ncols=1, figsize=pu.set_size(width, fraction=0.49), tight_layout=True,
)
# fig2, ax21 = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True,)
# ax22 = ax21.twinx()

cmap = matplotlib.cm.get_cmap("plasma", len(files_s) + 1)
color_lines = []
error_names = []
error_names_latex = []

reg_param_lines = []

for idx, (data_s, data_a) in enumerate(zip(data_symm, data_asymm)):
    # indici = np.abs(data_s.T[0]) <= 10.1
    # print(np.sum(indici))
    ax1.plot(
        data_s.T[0],
        data_s.T[1],
        linestyle="solid",
        color=cmap_disc(idx),
        label=r"$\beta$ = {:.2f}".format(beta[idx]),
    )

    indices = np.argsort(data_a.T[0])

    ax2.plot(
        np.exp(data_a.T[0][indices]),
        np.exp(data_a.T[1][indices]),
        linestyle="solid",
        # marker="P",
        color=cmap_disc(idx),
        label=r"$\beta$ = {:.2f}".format(beta[idx]),
    )

    # ax2.plot(
    #     data_a[0],
    #     data_a[1],
    #     linestyle="dashed",
    #     color=cmap(idx),
    #     label=r"$\beta$ = {:.2f}",
    # )

#  first_legend = ax.legend(loss_lines, ["L2", "Huber", "BayesOptimal"])
# ax.add_artist(first_legend)
# ax.legend(color_lines, error_names, loc="lower left")
ax1.set_ylabel(r"$\ell_{\mathrm{opt}}(0,z)$")
ax1.set_xlabel(r"$z$")
# ax1.set_xscale("log")
# ax1.set_yscale("log")
ax1.set_xlim([-10, 10])
ax1.set_ylim([0, 15])
# ax1.legend()  # prop={"size": 22}

# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# ax2.set_title(r"Optimal Loss")
ax2.set_ylabel(r"$\ell_{\mathrm{opt}}(0,z)$")
ax2.set_xlabel(r"$z$")
ax2.set_xscale("log")
ax2.set_yscale("log")
# # ax2.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
# # ax2.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
# # ax20.set_yscale("log")
# #  ax21.tick_params(axis='y', which='major', pad=7)
# #  ax21.set_yscale("log")
# # ax21.set_xlim([0.009, 110])
# ax2.legend()  # prop={"size": 22}

cbar = fig1.colorbar(
    sm,
    ticks=np.linspace(0, 1, 11),  #  (0.2, 2.1, step),  # np.logspace(-2, 2, 10),  #
    # boundaries=np.linspace(0, 11, 12),
    # norm=norm
    # format=ticker.FuncFormatter(myfmt),
)

cbar.ax.set_title(r"$\beta$")
cbar.ax.minorticks_off()

cbar2 = fig2.colorbar(
    sm,
    ticks=np.linspace(0, 1, 11),  #  (0.2, 2.1, step),  # np.logspace(-2, 2, 10),  #
    # boundaries=np.linspace(0, 11, 12),
    # norm=norm
    # format=ticker.FuncFormatter(myfmt),
)

cbar2.ax.set_title(r"$\beta$")
cbar2.ax.minorticks_off()

if save:
    pu.save_plot(fig1, "optimal_losses_double_noise_ds_0.1_dl_5.0_eps_0.1_different_beta")
    pu.save_plot(
        fig2, "optimal_losses_loglog_double_noise_ds_0.1_dl_5.0_eps_0.1_different_beta"
    )
plt.show()
