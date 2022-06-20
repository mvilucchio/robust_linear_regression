from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from src.utils import load_file
import src.plotting_utils as pu
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

save = True
experimental_points = True
width = 1.0 * 458.63788
random_number = np.random.randint(100)

marker_cycler = ["*", "s", "P", "P", "v", "D"]

deltas_large = [0.5, 1.0, 2.0, 5.0, 10.0]  # [0.5, 1.0, 2.0, 5.0, 10.0]  #
#  reg_params = [0.01, 0.1, 1.0, 10.0, 100.0]
percentages = [0.01, 0.05, 0.1, 0.3]  # 0.01, 0.05, 0.1, 0.3
betas = [0.0, 0.5, 1.0]

index_dl = 3
index_beta = 2
index_per = 3
dl = deltas_large[index_dl]
beta = betas[index_beta]
p = percentages[index_per]

# percentages.reverse()
#  for dl in deltas_large

L2_settings = [
    {
        "loss_name": "L2",
        "alpha_min": 0.01,
        "alpha_max": 100000,
        "alpha_pts": 100,
        "delta_small": 0.1,
        "delta_large": dl,
        "percentage": p,
        # "beta": beta,
        "experiment_type": "reg_param optimal",
    }
    for dl in deltas_large
]

L2_experimantal_settings = [
    {
        "loss_name": "L2",
        "alpha_min": 0.01,
        "alpha_max": 100000,
        "alpha_pts_theoretical": 150,
        "alpha_pts_experimental": 21,
        "delta_small": 0.1,
        "delta_large": dl,
        "percentage": p,
        "n_features": 1000,
        "repetitions": 10,
        # "beta": beta,
        "experiment_type": "reg_param optimal exp",
    }
    for dl in deltas_large
]

L1_settings = [
    {
        "loss_name": "L1",
        "alpha_min": 0.01,
        "alpha_max": 100000,
        "alpha_pts": 100,
        "delta_small": 0.1,
        "delta_large": dl,
        "percentage": p,
        # "beta": beta,
        "experiment_type": "reg_param optimal",
    }
    for dl in deltas_large
]
L1_experimantal_settings = [
    {
        "loss_name": "L1",
        "alpha_min": 0.01,
        "alpha_max": 100000,
        "alpha_pts_theoretical": 100,
        "alpha_pts_experimental": 21,
        "delta_small": 0.1,
        "delta_large": dl,
        "percentage": p,
        "n_features": 200,
        "repetitions": 5,
        # "beta": beta,
        "experiment_type": "reg_param optimal exp",
    }
    for dl in deltas_large
]

Huber_settings = [
    {
        "loss_name": "Huber",
        "alpha_min": 0.01,
        "alpha_max": 100000,
        "alpha_pts": 100,
        "delta_small": 0.1,
        "delta_large": dl,
        "percentage": p,
        # "beta": beta,
        "experiment_type": "reg_param huber_param optimal",
    }
    for dl in deltas_large
]

Huber_experimantal_settings = [
    {
        "loss_name": "Huber",
        "alpha_min": 0.01,
        "alpha_max": 100000,
        "alpha_pts_theoretical": 100,
        "alpha_pts_experimental": 21,
        "delta_small": 0.1,
        "delta_large": dl,
        "percentage": p,
        "n_features": 1000,
        "repetitions": 10,
        # "beta": beta,
        "experiment_type": "reg_param huber_param optimal exp",
    }
    for dl in deltas_large
]

BO_settings = [
    {
        "alpha_min": 0.01,
        "alpha_max": 1000,
        "alpha_pts": 46,
        "delta_small": 0.1,
        "delta_large": dl,
        "percentage": p,
        # "beta": beta,
        "experiment_type": "BO",
    }
    for dl in deltas_large
]

pu.initialization_mpl()

tuple_size = pu.set_size(width, fraction=0.49)

fig, ax = plt.subplots(1, 1, figsize=tuple_size)  # , tight_layout=True,
# fig.subplots_adjust(left=0.15)
# fig.subplots_adjust(bottom=0.0)
# fig.subplots_adjust(right=0.99)
fig.subplots_adjust(left=0.2)
fig.subplots_adjust(bottom=0.2)
fig.subplots_adjust(top=0.99)
fig.subplots_adjust(right=0.99)

figleg1 = plt.figure(figsize=(tuple_size[0], tuple_size[1] / 17))
figleg2 = plt.figure(figsize=(tuple_size[0], tuple_size[1] / 7))


fig2, (ax20, ax21) = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=tuple_size,
    gridspec_kw={"height_ratios": [1, 1]},
    # tight_layout=True,
)
fig2.subplots_adjust(left=0.2)
fig2.subplots_adjust(bottom=0.2)
fig2.subplots_adjust(top=0.99)
fig2.subplots_adjust(right=0.99)

# fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True,)
# fig2, (ax20, ax21) = plt.subplots(
#     nrows=2, ncols=1, sharex=True, figsize=(12, 9), gridspec_kw={"height_ratios": [1, 1]}
# )
# fig2.subplots_adjust(left=0.15)
# fig2.subplots_adjust(right=0.85)

# fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True,)
# fig2, ax21 = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True,)
# ax22 = ax21.twinx()

cmap = plt.get_cmap("tab10")
color_lines = []
error_names = []
error_names_latex = []

reg_param_lines = []

for idx, (L2_d, L2_e, L1_d, L1_e, Huber_d, Huber_e, BO_d) in enumerate(
    zip(
        L2_settings,
        L2_experimantal_settings,
        L1_settings,
        L1_experimantal_settings,
        Huber_settings,
        Huber_experimantal_settings,
        BO_settings,
    )
):
    alphas_L2, errors_L2, lambdas_L2 = load_file(**L2_d)
    alphas_L2_e, errors_mean_L2_e, errors_std_L2_e, _ = load_file(**L2_e)

    alphas_L1, errors_L1, lambdas_L1 = load_file(**L1_d)
    # alphas_L1_e, errors_mean_L1_e, errors_std_L1_e, _ = load_file(**L1_e)

    alphas_Huber, errors_Huber, lambdas_Huber, huber_params = load_file(**Huber_d)
    alphas_Huber_e, errors_mean_Huber_e, errors_std_Huber_e, _, _ = load_file(**Huber_e)

    alphas_BO, errors_BO = load_file(**BO_d)

    color_lines.append(Line2D([0], [0], color=cmap(idx)))
    error_names.append("Large Noise = ${:.1f}$".format(deltas_large[idx]))
    error_names_latex.append("$\Delta_\ell = {:.1f}$".format(deltas_large[idx]))

    ax.plot(
        alphas_L2, errors_L2, label=r"$\ell_2$", color=cmap(idx), linestyle="dotted"
    )  #
    ax.errorbar(
        alphas_L2_e,
        errors_mean_L2_e,
        errors_std_L2_e,
        color=cmap(idx),
        linestyle="None",
        marker=".",
        markersize=1.0,
    )

    ax.plot(
        alphas_L1, errors_L1, label=r"$\ell_1$", color=cmap(idx), linestyle="dashdot",
    )
    # ax.errorbar(
    #     alphas_L1_e,
    #     errors_mean_L1_e,
    #     errors_std_L1_e,
    #     color=cmap(idx),
    #     linestyle="None",
    #     marker=".",
    #     markersize=1.0,
    # )

    ax.plot(
        alphas_Huber, errors_Huber, label="Huber", color=cmap(idx), linestyle="dashed",
    )
    ax.errorbar(
        alphas_Huber_e,
        errors_mean_Huber_e,
        errors_std_Huber_e,
        color=cmap(idx),
        linestyle="None",
        marker=".",
        markersize=1.0,
    )

    alphas_BO, errors_BO = load_file(**BO_settings[0])
    ax.plot(
        alphas_BO, errors_BO, label="BO", color=cmap(idx), linestyle="solid",
    )

    ax20.plot(
        alphas_L2,
        lambdas_L2,
        label=r"$\ell_2$",
        linestyle="dotted",
        # marker="P",
        color=cmap(idx),
        # marker=marker_cycler[idx],
        # linewidth=1.0,
        # color='b',
        # label="$\Delta_\ell = {:.1f} \:\epsilon = {:.2f}$ (L2)".format(*all_params[idx])
    )

    ax20.plot(
        alphas_L1,
        lambdas_L1,
        label=r"$\ell_1$",
        linestyle="dashdot",  #  ("dashdotted", (0, (3, 5, 1, 5))),
        # marker="P",
        color=cmap(idx),
        # marker=marker_cycler[idx],
        # linewidth=1.0,
        # color='b',
        # label="$\Delta_\ell = {:.1f} \:\epsilon = {:.2f}$ (L2)".format(*all_params[idx])
    )

    ax20.plot(
        alphas_Huber,
        lambdas_Huber,
        label="Huber",
        linestyle="dashed",
        # color=cmap(idx),
        # marker=marker_cycler[idx],
        # linewidth=1.0,
        # color='b',
        # label="$\Delta_\ell = {:.1f} \:\epsilon = {:.2f}$ (Huber)".format(*all_params[idx])
    )

    ax21.plot(
        alphas_Huber,
        huber_params,
        label="Huber",
        linestyle="dashed",
        # color=cmap(idx),
        # marker=marker_cycler[idx],
        # linewidth=1.0,
        #  color='g',
        # label="$\Delta_\ell = {:.1f} \:\epsilon = {:.2f}$ (Huber)".format(*all_params[idx])
    )

    # ax3.plot(
    #     alphas_L2,
    #     (errors_L2 - errors_BO) / errors_BO,
    #     linestyle="dotted",
    #     color=cmap(idx),
    # )

    # ax3.plot(
    #     alphas_Huber,
    #     (errors_Huber - errors_BO) / errors_BO,
    #     linestyle="dashed",
    #     color=cmap(idx),
    # )

loss_lines = [
    Line2D([0], [0], color="k", linestyle="dotted"),
    Line2D([0], [0], color="k", linestyle="dashdot"),
    Line2D([0], [0], color="k", linestyle="dashed"),
    Line2D([0], [0], color="k", linestyle="solid"),
]

loss_lines_parameters = [
    Line2D([0], [0], color="k", linestyle="dotted"),
    Line2D([0], [0], color="k", linestyle="dashdot"),
    Line2D([0], [0], color="k", linestyle="dashed"),
]

loss_lines_relative = [
    Line2D([0], [0], color="k", linestyle="dotted"),
    Line2D([0], [0], color="k", linestyle="dashdot"),
    Line2D([0], [0], color="k", linestyle="dashed"),
]

figleg1.legend(loss_lines, [r"$\ell_2$", r"$\ell_1$", "Huber", "BO"], "center", ncol=4)
figleg2.legend(color_lines, error_names_latex, "center", ncol=3, handlelength=2.1)

# first_legend = ax.legend(
#     loss_lines,
#     [r"$\ell_2$", r"$\ell_1$", "Huber", "BO"],
#     loc="lower left",
#     bbox_to_anchor=(0.0, 0.25),
# )
# ax.add_artist(first_legend)
# ax.legend(
#     color_lines, error_names_latex, loc="lower left", bbox_to_anchor=(0.0, 0.0),
# )

ax.set_ylabel(
    r"$\frac{1}{d} \mathbb{E}[\norm{\bf{\hat{w}} - \bf{w^\star}}^2]$"  # Generalization Error:
)
ax.set_xlabel(r"$\alpha$")
ax.set_xscale("log")
ax.set_yscale("log")
# ax.set_xlim([0.009, 110])
# ax.legend()
# ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

fig2.subplots_adjust(wspace=0, hspace=0)
ax20.set_ylabel(r"$\lambda_{\mathrm{opt}}$")  # "Opt. Reg.\nParameter"
ax21.set_ylabel(r"$a_{\mathrm{opt}}$")  # "Opt. Huber\nParameter"
ax21.set_xlabel(r"$\alpha$")
ax20.set_xscale("log")
ax20.tick_params(axis="y", which="major", pad=5)
# ax20.set_yscale("log")
ax20.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax21.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax21.tick_params(axis="y", which="major", pad=5)
# ax21.set_yscale("log")
# ax21.set_xlim([100, 100000])
# ax20.legend()
# ax21.legend()
# ax20.legend()

# box = ax20.get_position()
# ax20.set_position([box.x0, box.y0, box.width * 0.9, box.height])
# box2 = ax21.get_position()
# ax21.set_position([box2.x0, box2.y0, box2.width * 0.9, box2.height])

# second_legend = ax21.legend(
#     loss_lines_relative,
#     [r"$\ell_2$", r"$\ell_1$", "Huber"],
#     loc="upper left",
#     bbox_to_anchor=(0.5, 1.0),
# )
# ax21.add_artist(second_legend)
# ax20.legend(color_lines, error_names_latex, loc="upper left", bbox_to_anchor=(0.45, 1.0))


# third_legend = ax3.legend(loss_lines_relative, ["L2", "Huber"])
# ax3.add_artist(third_legend)
# ax3.legend(color_lines, error_names, loc="lower right")

# ax3.set_ylabel("Generalization Error Relative to Bayes Optimal")
# ax3.set_xlabel(r"$\alpha$")
# ax3.set_xscale("log")
# ax3.set_yscale("log")
# ax21.set_xlim([0.009, 110])


if save:
    # pu.save_plot(
    #     fig,
    #     "total_optimal_confronts_fixed_percentage_{:.2f}_beta_{:.2f}".format(
    #         percentages[index_per], betas[index_beta]
    #     ),
    # )
    pu.save_plot(
        fig2,
        "total_optimal_confronts_params_fixed_percentage_{:.2f}_beta_{:.2f}".format(
            percentages[index_per], betas[index_beta]
        ),
    )
    # pu.save_plot(
    #     figleg1,
    #     "total_optimal_confronts_leg1_fixed_percentage_{:.2f}_beta_{:.2f}".format(
    #         percentages[index_per], betas[index_beta]
    #     ),
    # )
    # pu.save_plot(
    #     figleg2,
    #     "total_optimal_confronts_leg2_fixed_percentage_{:.2f}_beta_{:.2f}".format(
    #         percentages[index_per], betas[index_beta]
    #     ),
    # )
    # pu.save_plot(
    #     fig3,
    #     "total_optimal_confronts_relative_fixed_percentage_{:.2f}".format(
    #         deltas_large[0]
    #     ),
    # )

plt.show()
