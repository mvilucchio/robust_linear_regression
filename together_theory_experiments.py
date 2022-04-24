import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from src.utils import check_saved, load_file, save_file, experiment_runner

random_number = np.random.randint(0, 100)

names_cm = ["Purples", "Blues", "Greens", "Oranges", "Greys"]


def get_cmap(n, name="hsv"):
    return plt.cm.get_cmap(name, n)


loss_name = "L2"
delta_small, delta_large, percentage = 0.1, 10.0, 0.3
reg_params = [0.01, 0.1, 1.0]

experimental_settings = [
    {
        "loss_name": loss_name,
        "alpha_min": 0.01,
        "alpha_max": 100,
        "alpha_pts": 13,
        "reg_param": reg_param,
        "repetitions": 10,
        "n_features": 500,
        "delta_small": delta_small,
        "delta_large": delta_large,
        "percentage": percentage,
        "a": 0.5,
        "experiment_type": "exp",
    }
    for reg_param in reg_params
]

theory_settings = [
    {
        "loss_name": loss_name,
        "alpha_min": 0.01,
        "alpha_max": 100,
        "alpha_pts": 23,
        "reg_param": reg_param,
        "delta_small": delta_small,
        "delta_large": delta_large,
        "percentage": percentage,
        "a": 0.5,
        "experiment_type": "theory",
    }
    for reg_param in reg_params
]

n_exp = len(theory_settings)

alphas_num = [None] * n_exp
errors_mean_num = [None] * n_exp
errors_std_num = [None] * n_exp

alphas_theory = [None] * n_exp
errors_theory = [None] * n_exp

for idx, (exp_dict, theory_dict) in enumerate(
    zip(tqdm(experimental_settings, desc="reg_param", leave=False), theory_settings)
):
    file_exists, file_path = check_saved(**exp_dict)

    if not file_exists:
        experiment_runner(**exp_dict)

    exp_dict.update({"file_path": file_path})
    alphas_num[idx], errors_mean_num[idx], errors_std_num[idx] = load_file(**exp_dict)

    file_exists, file_path = check_saved(**theory_dict)

    if not file_exists:
        experiment_runner(**theory_dict)

    theory_dict.update({"file_path": file_path})
    alphas_theory[idx], errors_theory[idx] = load_file(**theory_dict)

# ------------

fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)

for idx, (al_n, err_m, err_s, al_t, err_t) in enumerate(
    zip(alphas_num, errors_mean_num, errors_std_num, alphas_theory, errors_theory)
):
    colormap = get_cmap(n_exp + 3)
    ax.plot(
        al_t,
        err_t,
        # marker='.',
        label=r"$\lambda = {}$".format(reg_params[idx]),
        color=colormap(idx + 3),
    )

    ax.errorbar(
        al_n, err_m, err_s, marker=".", linestyle="None", color=colormap(idx + 3),
    )

ax.set_title(
    r"{} Loss - $\Delta = [{:.2f}, {:.2f}], \epsilon = {:.2f}$".format(
        loss_name, delta_small, delta_large, percentage
    )
)
ax.set_ylabel(r"$\frac{1}{d} E[||\hat{w} - w^\star||^2]$")
ax.set_xlabel(r"$\alpha$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([0.009, 110])
ax.minorticks_on()
ax.grid(True, which="both")
ax.legend()

fig.savefig(
    "./imgs/{} - together - double noise - {}.png".format(loss_name, random_number),
    format="png",
    dpi=150,
)

plt.show()
