from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from src.utils import load_file
import src.plotting_utils as pu
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from itertools import product
from src.optimal_lambda import (
    optimal_lambda,
    optimal_reg_param_and_huber_parameter,
    no_parallel_optimal_reg_param_and_huber_parameter,
)

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import src.plotting_utils as pu

from scipy.optimize import minimize
import src.fpeqs as fp
from src.fpeqs_BO import (
    var_func_BO,
    var_hat_func_BO_single_noise,
    var_hat_func_BO_num_double_noise,
    var_hat_func_BO_num_decorrelated_noise,
)
from src.fpeqs_L2 import (
    var_func_L2,
    var_hat_func_L2_single_noise,
    var_hat_func_L2_double_noise,
    var_hat_func_L2_decorrelated_noise,
)
from src.fpeqs_L1 import (
    var_hat_func_L1_single_noise,
    var_hat_func_L1_double_noise,
    var_hat_func_L1_decorrelated_noise,
)
from src.fpeqs_Huber import (
    var_hat_func_Huber_single_noise,
    var_hat_func_Huber_double_noise,
    var_hat_func_Huber_decorrelated_noise,
)

SMALLEST_REG_PARAM = 1e-7
SMALLEST_HUBER_PARAM = 1e-7
MAX_ITER = 2500
XATOL = 1e-8
FATOL = 1e-8

save = True
experimental_points = True
width = 0.4 * 458.63788

delta_large = 5.0
beta = 1.0
p = 0.1
delta_small = 1.0

pu.initialization_mpl()

tuple_size = pu.set_size(width, fraction=0.50)
# tuple_size = pu.set_size_square(width, fraction=0.50)

fig, ax = plt.subplots(1, 1, figsize=tuple_size, zorder=30)
# fig, ax = plt.subplots(1, 1, figsize=(width,width), zorder=30)
fig.subplots_adjust(left=0.35)
fig.subplots_adjust(bottom=0.35)
fig.subplots_adjust(top=0.95)
fig.subplots_adjust(right=0.95)

cmap = plt.get_cmap("tab10")
color_lines = []
error_names = []
error_names_latex = []

reg_param_lines = []

while True:
    m = 0.89 * np.random.random() + 0.1
    q = 0.89 * np.random.random() + 0.1
    sigma = 0.89 * np.random.random() + 0.1
    if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
        initial_condition = [m, q, sigma]
        break

# alphas_L2, errors_L2, lambdas_L2 = load_file(**L2_settings)

pup = {
    "delta_small": delta_small,
    "delta_large": delta_large,
    "percentage": p,
    "beta": beta,
}

alphas_L2, errors_L2, lambdas_L2 = optimal_lambda(
    var_func_L2,
    var_hat_func_L2_decorrelated_noise,
    alpha_1=0.1,
    alpha_2=10000,
    n_alpha_points=100,
    initial_cond=initial_condition,
    var_hat_kwargs=pup,
)

alphas_L1, errors_L1, lambdas_L1 = optimal_lambda(
    var_func_L2,
    var_hat_func_L1_decorrelated_noise,
    alpha_1=0.1,
    alpha_2=10000,
    n_alpha_points=100,
    initial_cond=initial_condition,
    var_hat_kwargs=pup,
)

# alphas_Huber, errors_Huber, lambdas_Huber, huber_params = load_file(**Huber_settings)
pep = {
    "delta_small": delta_small,
    "delta_large": delta_large,
    "percentage": p,
    "beta": beta,
}

(
    alphas_Huber,
    errors_Huber,
    lambdas_Huber,
    huber_params,
) = no_parallel_optimal_reg_param_and_huber_parameter(
    var_hat_func=var_hat_func_Huber_decorrelated_noise,
    alpha_1=0.1,
    alpha_2=10000,
    n_alpha_points=100,
    initial_cond=initial_condition,
    var_hat_kwargs=pep,
)

pap = {
    "delta_small": delta_small,
    "delta_large": delta_large,
    "percentage": p,
    "beta": beta,
}

alphas_BO, (errors_BO,) = fp.no_parallel_different_alpha_observables_fpeqs(
    var_func_BO,
    var_hat_func_BO_num_decorrelated_noise,
    alpha_1=0.1,
    alpha_2=10000,
    n_alpha_points=100,
    initial_cond=initial_condition,
    var_hat_kwargs=pap,
)

# alphas_BO, errors_BO = load_file(**BO_settings)

dat_l2_hub = np.genfromtxt(
    "./data/GOOD_beta_1.0_all.csv",  # "./data/numerics_l2_sweep_alpha_fixed_eps_0.30_beta_0.00_delta_large_5.00_delta_small_1.00_dim_1000.00_bak.csv",
    skip_header=1,
    delimiter=",",
)
alph_num = dat_l2_hub[:, 0]
err_mean_l2 = dat_l2_hub[:, 1]
err_std_l2 = dat_l2_hub[:, 2]
err_mean_l1 = dat_l2_hub[:, 3]
err_std_l1 = dat_l2_hub[:, 4]
err_mean_hub = dat_l2_hub[:, 5]
err_std_hub = dat_l2_hub[:, 6]

# dat_l1 = np.genfromtxt(
#     "./data/GOOD_beta_1.0_l1.csv", # "./data/numerics_sweep_alpha_just_l1_fixed_eps_0.30_beta_0.00_delta_large_5.00_delta_small_1.00_dim_500.00_bak.csv",
#     skip_header=1,
#     delimiter=",",
#     # dtype="float",
# )
# alpha_l1 = dat_l1[:, 0]
# err_mean_l1 = dat_l1[:, 1]
# err_std_l1 = dat_l1[:, 2]


ax.plot(
    alphas_L2,
    errors_L2-errors_BO,
    label=r"$\ell_2$",
    color="tab:blue",
    zorder=3,  # ,linewidth=1.0
)

# ax.errorbar(
#     alph_num,
#     err_mean_l2,
#     yerr=err_std_l2,
#     color="tab:blue",
#     linestyle="",
#     elinewidth=0.75,
#     markerfacecolor="none",
#     markeredgecolor="tab:blue",
#     marker="o",
#     markersize=1.0,
#     zorder=3
# )

ax.plot(
    alphas_L1,
    errors_L1-errors_BO,
    label=r"$\ell_1$",
    color="tab:green",
    zorder=5
    # linewidth=1.0
)
# ax.errorbar(
#     alph_num,
#     err_mean_l1,
#     yerr=err_std_l1,
#     color="tab:green",
#     linestyle="",
#     elinewidth=0.75,
#     markerfacecolor="none",
#     markeredgecolor="tab:green",
#     marker="o",
#     markersize=1.0,
#     zorder=5
# )

ax.plot(
    alphas_Huber,
    errors_Huber-errors_BO,
    label="Huber",
    color="tab:orange",
    zorder=10,  # , linewidth=1.0
)  # r"$\mathcal{L}_{a_{\text{\tiny{opt}}}}$",

# ax.errorbar(
#     alph_num,
#     err_mean_hub,
#     yerr=err_std_hub,
#     color="tab:orange",
#     linestyle="",
#     elinewidth=0.75,
#     markerfacecolor="none",
#     markeredgecolor="tab:orange",
#     marker="o",
#     markersize=1.0,
#     zorder=10
# )


# alphas_BO, errors_BO = load_file(**BO_settings)

# ax.plot(
#     alphas_BO,
#     errors_BO,
#     label="BO",
#     color="tab:red",
#     linewidth=0.5,
#     zorder=15
# )

# np.savetxt(
#     "./data/sweep_alpha_fixed_eps_{:.2f}_beta_{:.2f}_delta_large_{:.2f}_delta_small.csv".format(
#         p, beta, delta_large, delta_small
#     ),
#     np.vstack(
#         (
#             alphas_L2,
#             errors_L2,
#             lambdas_L2,
#             errors_L1,
#             lambdas_L1,
#             errors_Huber,
#             lambdas_Huber,
#             huber_params,
#             errors_BO,
#         )
#     ).T,
#     delimiter=",",
#     header="delta_large,l2,l1,Huber,BO",
# )

# ax.plot(alph_num, err_mean_l2, color="tab:blue", linestyle="", markersize=1, marker=".")
# ax.plot(alph_num, err_mean_l1, color="tab:orange", linestyle="", markersize=1, marker=".")
# ax.plot(alph_num, err_mean_hub, color="tab:green", linestyle="", markersize=1, marker=".")

# plt.axvline(x=10, ymin=0.0, ymax=1.0, color="black", alpha=0.7, linewidth=1, linestyle="--")

# xytext=(1,1), textcoords="offset points",
# ax.annotate(
#     xy=(1100,0.5), text=r"$\ell_2$", va="center", color='blue',
# )
# ax.annotate(
#     xy=(1100,0.05), text=r"$\ell_1$", va="center", color='orange',
# )
# ax.annotate(
#     xy=(200,0.15), text="Huber", va="center", color='green',
# )
# ax.annotate(
#     xy=(300,0.002), text="BO", va="center", color='red',
# )
# ax.annotate(xy=(alphas_L1[-1],errors_L1[-1]), xytext=(5,0), textcoords='offset points', text="$\ell_1$", va='center')
# ax.annotate(xy=(alphas_Huber[-1],errors_Huber[-1]), xytext=(5,0), textcoords='offset points', text="Huber", va='center')
# ax.annotate(xy=(errors_BO[-1],errors_BO[-1]), xytext=(5,0), textcoords='offset points', text=r"BO", va='center')

# ax.set_ylabel(r"$E_{\text{gen}}$")
# ax.set_ylabel(r"$E_{\text{gen}}-E_{\text{gen}}^{\text{BO}}$")
ax.set_ylabel(r"\tiny{$\Delta E_{\text{gen}}$}", labelpad=1.0)
ax.set_xlabel(r"\tiny{$\alpha$}", labelpad=-4.0)
# ax.set_ylabel(r"$\Delta E_{\text{gen}}$", labelpad=2.0)
# ax.set_xlabel(r"$\alpha$", labelpad=-2.0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([0.1, 100])
ax.set_ylim([0.00001, 0.1])
# ax.set_ylim([0.0, 0.08])
ax.grid(zorder=20)
ax.zorder = 20
# ax.legend(loc="best", bbox_to_anchor=(1, 0.5))

# labels = [item.get_text() for item in ax.get_xticklabels()]
# print(labels)
# # labels[2] =
# # ax.set_xticklabels([0.1, 1, 10, 100, 10000], [r"10^{-1}",r"10^{0}",r"$\alpha_{\text{cut}}$",r"10^{2}",r"10^{3}"])
# text = [m.get_text() for m in ax.get_xticklabels()]
# positions = [m for m in ax.get_xticks()]

# print(text)
# print(positions)
# ax.ticklabel_format(axis='x',scilimits=(0,0))
# fmt = matplotlib.ticker.StrMethodFormatter(r"\tiny{x}")
# # ax.xaxis.set_major_formatter(fmt)
# ax.yaxis.set_major_formatter(fmt)

ax.set_xticks([0.1, 1, 10, 100])
ax.set_xticklabels([r"\tiny{$10^{-1}$}", r"\tiny{$10^{0}$}", r"\tiny{$10^{1}$}", r"\tiny{$10^{2}$}"])

# ax.set_yticks([0.0, 0.04, 0.08])
# ax.set_yticklabels([r"\tiny{$0.0$}", r"\tiny{$0.04$}", r"\tiny{$0.08$}"])

ax.set_yticks([0.00001, 0.001, 0.1])
ax.set_yticklabels([r"\tiny{$10^{-5}$}", r"\tiny{$10^{-3}$}", r"\tiny{$10^{-1}$}"])

# ax.set_xticks([0.1, 1, 10, 100])
# ax.set_xticklabels([r"{$10^{-1}$}", r"{$10^{0}$}", r"{$10^{1}$}", r"{$10^{2}$}"])

# ax.set_yticks([0.0, 0.04, 0.08])
# ax.set_yticklabels([r"{$0.0$}", r"{$0.04$}", r"{$0.08$}"])

ax.tick_params(axis="y", pad=1.0)
ax.tick_params(axis="x", pad=1.0)

# for tick in ax.get_xaxis().get_major_ticks():
#     tick.set_pad(4.)
#     # tick.label1 = tick._get_text1()

# for tick in ax.get_yaxis().get_major_ticks():
#     tick.set_pad(4.)
#     # tick.label1 = tick._get_text1()

if save:
    pu.save_plot(
        fig,
        "presentation_total_optimal_confronts_inset_fixed_delta_{:.2f}_beta_{:.2f}_delta_small_{:.2f}_eps_{:.2f}".format(
            delta_large, beta, delta_small, p
        ),
    )

plt.show()
