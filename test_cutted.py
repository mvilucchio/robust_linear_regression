import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import src.plotting_utils as pu
from src.numerics import (
    find_coefficients_cutted_l2,
    generate_different_alpha,
    no_parallel_generate_different_alpha,
)
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

    alphas, errors_mean, errors_std = no_parallel_generate_different_alpha(
        num.measure_gen_decorrelated,
        num.find_coefficients_double_quad,
        alpha_1=5,
        alpha_2=500,
        n_features=200,
        n_alpha_points=4,
        repetitions=8,
        reg_param=0.0,
        measure_fun_kwargs={
            "delta_small": 0.1,
            "delta_large": 10.0,
            "percentage": 0.1,
            "beta": 0.5,
        },
        find_coefficients_fun_kwargs={"a": 0.25},
        alphas=None,
    )

    np.savez(
        "./double_quad_loss_beta_{}_eps_{}_dl_{}_ds_{}_rp_{}_a_{}".format(
            0.5, 0.1, 10.0, 0.1, 0.0, 0.25
        ),
        alphas=alphas,
        errors_mean=errors_mean,
        errors_std=errors_std,
    )
    # dat = np.load(
    #     "./double_quad_loss_beta_{}_eps_{}_dl_{}_ds_{}.npz".format(0.5, 0.1, 10.0, 0.1)
    # )
    # alphas = dat["alphas"]
    # errors_mean = dat["errors_mean"]
    # errors_std = dat["errors_std"]

    experiment_runner(
        **{
            # "loss_name": "Huber",
            "alpha_min": 0.01,
            "alpha_max": 10000,
            "alpha_pts": 100,
            #  "reg_param": 0.5,
            # "delta" : delta_large,
            "delta_small": 0.1,
            "delta_large": 10.0,
            "percentage": 0.3,
            "beta": 0.5,
            # "a": 1.0,
            "experiment_type": "reg_param huber_param optimal",
        }
    )

    alp_H, err_H, _, _ = load_file(
        **{
            # "loss_name": "Huber",
            "alpha_min": 0.01,
            "alpha_max": 10000,
            "alpha_pts": 100,
            #  "reg_param": 0.5,
            # "delta" : delta_large,
            "delta_small": 0.1,
            "delta_large": 10.0,
            "percentage": 0.3,
            "beta": 0.5,
            # "a": 1.0,
            "experiment_type": "reg_param huber_param optimal",
        }
    )

    experiment_runner(
        **{
            "alpha_min": 0.1,
            "alpha_max": 1000,
            "alpha_pts": 50,
            "reg_param": 0.5,
            "delta_small": 0.1,
            "delta_large": 10.0,
            "percentage": 0.3,
            "beta": 0.5,
            "experiment_type": "BO",
        }
    )

    alp, er = load_file(
        **{
            "alpha_min": 0.1,
            "alpha_max": 1000,
            "alpha_pts": 50,
            "reg_param": 0.5,
            "delta_small": 0.1,
            "delta_large": 10.0,
            "percentage": 0.3,
            "beta": 0.5,
            "experiment_type": "BO",
        }
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
    ax.plot(alp, er, label="BO")
    ax.plot(alp_H, err_H, label="Huber")

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
