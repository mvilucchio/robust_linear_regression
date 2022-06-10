import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import src.plotting_utils as pu
from src.numerics import find_coefficients_cutted_l2, generate_different_alpha
import src.numerics as num
from src.utils import (
    check_saved,
    load_file,
    save_file,
    experiment_runner,
)

save = False

if __name__ == "__main__":
    random_number = np.random.randint(0, 100)

    names_cm = ["Purples", "Blues", "Greens", "Oranges", "Greys"]

    def get_cmap(n, name="hsv"):
        return plt.cm.get_cmap(name, n)

    loss_name = "L1"
    delta_small, delta_large, percentage, beta = 0.1, 5.0, 0.1, 0.0
    reg_params = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]

    experimental_settings = [
        {
            "loss_name": loss_name,
            "alpha_min": 0.01,
            "alpha_max": 100,
            "alpha_pts": 36,
            "reg_param": reg_param,
            "repetitions": 20,
            "n_features": 1000,
            # "delta" : delta_large,
            "delta_small": delta_small,
            "delta_large": delta_large,
            "percentage": percentage,
            "beta": beta,
            # "a": 1.0,
            "experiment_type": "exp",
        }
        for reg_param in reg_params
    ]

    theory_settings = [
        {
            "loss_name": loss_name,
            "alpha_min": 0.1,
            "alpha_max": 100000,
            "alpha_pts": 100,
            "reg_param": reg_param,
            # "delta" : delta_large,
            "delta_small": delta_small,
            "delta_large": delta_large,
            "percentage": percentage,
            "beta": beta,
            "a": 1.0,
            "experiment_type": "theory",
        }
        for reg_param in reg_params
    ]

    n_exp = len(theory_settings)

    alphas, errors_mean, errors_std = generate_different_alpha(
        num.measure_gen_decorrelated,
        num.find_coefficients_cutted_l2,
        alpha_1=0.01,
        alpha_2=10,
        n_features=200,
        n_alpha_points=13,
        repetitions=5,
        reg_param=0.5,
        measure_fun_kwargs={
            "delta_small": 0.1,
            "delta_large": 10.0,
            "percentage": 0.1,
            "beta": 0.0,
        },
        find_coefficients_fun_kwargs={"a": 1.0},
        alphas=None,
    )

    # ------------

    pu.initialization_mpl()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)
    # ax.plot(
    #     al_t,
    #     err_t,
    #     # linewidth=1,
    #     # marker='.',
    #     label=r"$\lambda = {}$".format(reg_params[idx]),
    #     color=colormap(idx + 3),
    # )

    ax.errorbar(alphas, errors_mean, errors_std, marker=".", linestyle="None")

    # ax.set_title(
    #     r"{} Loss - $\Delta = [{:.2f}, {:.2f}], \epsilon = {:.2f}$".format(
    #         loss_name, delta_small, delta_large, percentage
    #     )
    # )
    ax.set_ylabel(
        r"Generalization Error: $\frac{1}{d} \mathbb{E}\qty[\norm{\bf{\hat{w}} - \bf{w^\star}}^2]$"
    )
    ax.set_xlabel(r"$\alpha$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_xlim([0.009, 110])
    ax.grid()
    ax.legend(prop={"size": 22})

    if save:
        pu.save_plot(
            fig,
            "{}_1.0_theory_experiment_comparison_decorrelated_ds_{}_dl_{}_eps_{}_beta_{}".format(
                loss_name, delta_small, delta_large, percentage, beta
            ),
        )

    plt.show()
