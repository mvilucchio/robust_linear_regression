from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from src.utils import load_file
import src.plotting_utils as pu
import matplotlib.pyplot as plt
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

save = False
experimental_points = True
width = 1.0 * 458.63788
# width = 398.3386
random_number = np.random.randint(100)

marker_cycler = ["*", "s", "P", "P", "v", "D"]

# deltas_large = [0.5, 1.0, 2.0, 5.0, 10.0]  # [0.5, 1.0, 2.0, 5.0, 10.0]  #
# # reg_params = [0.01, 0.1, 1.0, 10.0, 100.0]
# percentages = [0.05, 0.1, 0.3]  # 0.01, 0.05, 0.1, 0.3
# betas = [0.0, 0.5, 1.0]

# index_dl = 3
# index_beta = 2
# index_per = 2


delta_large = 5.0
beta = 0.0
p = 0.3
delta_small = 1.0

pu.initialization_mpl()

tuple_size = pu.set_size(width, fraction=0.49)

fig, ax = plt.subplots(1, 1, figsize=tuple_size)  # , tight_layout=True,
# fig.subplots_adjust(left=0.15)
# fig.subplots_adjust(bottom=0.0)
# fig.subplots_adjust(right=0.99)
fig.subplots_adjust(left=0.2)
fig.subplots_adjust(bottom=0.2)
fig.subplots_adjust(top=0.99)
fig.subplots_adjust(right=0.96)

# figleg1 = plt.figure(figsize=(tuple_size[0], tuple_size[1] / 17))
# figleg2 = plt.figure(figsize=(tuple_size[0], tuple_size[1] / 17))

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

# alphas_BO, errors_BO = load_file(**BO_settings)

ax.plot(alphas_L2, errors_L2, label=r"$\ell_2$")  #

ax.plot(
    alphas_L1,
    errors_L1,
    label=r"$\ell_1$",
)

ax.plot(alphas_Huber, errors_Huber, label="Huber")  # r"$\mathcal{L}_{a_{\text{\tiny{opt}}}}$",

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
    alpha_2=1000,
    n_alpha_points=60,
    initial_cond=initial_condition,
    var_hat_kwargs=pap,
)

# alphas_BO, errors_BO = load_file(**BO_settings)

ax.plot(
    alphas_BO,
    errors_BO,
    label="BO",
)

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

dat_l2_hub = np.genfromtxt(
    "./data/numerics_l2_sweep_alpha_fixed_eps_0.30_beta_0.00_delta_large_5.00_delta_small_1.00_dim_1000.00_bak.csv",
    skip_header=1,
    delimiter=",",
    # dtype="float",
)
alph_num = dat_l2_hub[:,0]
err_mean_l2 = dat_l2_hub[:,1]
err_std_l2 = dat_l2_hub[:,2]
err_mean_hub = dat_l2_hub[:,3]
err_std_hub = dat_l2_hub[:,4]

dat_l1 = np.genfromtxt(
    "./data/numerics_sweep_alpha_just_l1_fixed_eps_0.30_beta_0.00_delta_large_5.00_delta_small_1.00_dim_500.00_bak.csv",
    skip_header=1,
    delimiter=",",
    # dtype="float",
)
alpha_l1 = dat_l1[:,0]
err_mean_l1 = dat_l1[:,1]
err_std_l1 = dat_l1[:,2]

ax.errorbar(alph_num, err_mean_l2, yerr=err_std_l2, label="l2")
ax.errorbar(alpha_l1, err_mean_l1, yerr=err_std_l1, label="l1")
ax.errorbar(alph_num, err_mean_hub, yerr=err_std_hub, label="h")

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

ax.set_ylabel(r"$E_{\text{gen}}$")
# r"$\frac{1}{d} \mathbb{E}[\norm{\bf{\hat{w}} - \bf{w^\star}}^2]$"  #
ax.set_xlabel(r"$\alpha$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([0.1, 10000])
ax.set_ylim([0.01, 1.5])
ax.legend()
# ax.legend(loc="best", bbox_to_anchor=(1, 0.5))

# labels = [item.get_text() for item in ax.get_xticklabels()]
# print(labels)
# # labels[2] =
# # ax.set_xticklabels([0.1, 1, 10, 100, 10000], [r"10^{-1}",r"10^{0}",r"$\alpha_{\text{cut}}$",r"10^{2}",r"10^{3}"])
# text = [m.get_text() for m in ax.get_xticklabels()]
# positions = [m for m in ax.get_xticks()]

# print(text)
# print(positions)

# ax.set_xticks([0.1, 1, 10, 100, 10000])
# ax.set_xticklabels([r"$10^{-1}$", r"$10^{0}$", r"$\alpha_{\text{cut}}$", r"$10^{2}$", r"$10^{3}$"])


if save:
    pu.save_plot(
        fig,
        "presentation_total_optimal_confronts_fixed_delta_{:.2f}_beta_{:.2f}".format(
            delta_large, beta
        ),
    )

plt.show()
