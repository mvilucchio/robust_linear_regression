from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
import src.plotting_utils as pu
from itertools import product
from matplotlib.lines import Line2D

save = True


def pout_double_gaussians(y, z, eps, delta_small, delta_large, beta):
    return (1 - eps) / (np.sqrt(2 * np.pi * delta_small)) * np.exp(
        -0.5 * (y - z) ** 2 / (delta_small)
    ) + eps / (np.sqrt(2 * np.pi * delta_large)) * np.exp(
        -0.5 * (y - beta * z) ** 2 / (delta_large)
    )


if __name__ == "__main__":
    pu.initialization_mpl()

    z = 1.0

    ys = np.linspace(-3, 5, 500)
    delta_small = 0.1
    delta_larges = [0.5, 1.0, 2.0, 5.0, 10.0]
    percentages = [0.01, 0.05, 0.1, 0.3]
    beta = 0.5

    fig, ax = plt.subplots(5, 1, figsize=(7, 8), sharex=True)
    fig.subplots_adjust(hspace=0.5, right=0.75)
    # fig.tight_layout()
    cmap = plt.get_cmap("tab10")

    color_lines = []
    error_names = []
    error_names_latex = []

    for idx, dl in enumerate(delta_larges):
        for jdx, eps in enumerate(percentages):
            if idx == 0:
                color_lines.append(Line2D([0], [0], color=cmap(jdx)))
                error_names.append("Large Noise = ${:.2f}$".format(eps))
                error_names_latex.append("$\epsilon = {:.2f}$".format(eps))
            ax[idx].plot(
                ys,
                pout_double_gaussians(ys, z, eps, delta_small, dl, beta),
                color=cmap(jdx),
            )
        # ax[idx].set_ylim([0, 1.5])
        # ax[idx].legend()
        ax[idx].set_title("$\Delta_\ell = {:.2f}$".format(dl), size=22)
        ax[idx].set_yscale("log")

    ax[0].legend(
        color_lines, error_names_latex, loc="lower left", bbox_to_anchor=(1, -3),
    )

    if save:
        pu.save_plot(fig, "pout_pdfs_log_ds_0.1_beta_{:.2f}".format(beta))

    plt.show()
