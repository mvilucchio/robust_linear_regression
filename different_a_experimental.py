import numpy as np
import matplotlib.pyplot as plt
import src.fpeqs as fpe
import src.numerics as num
import src.plotting_utils as pu
from tqdm.auto import tqdm
from src.utils import check_saved, load_file, save_file, experiment_runner

save = True
random_number = np.random.randint(0, 100)

names_cm = ["Purples", "Blues", "Greens", "Oranges", "Greys"]

def get_cmap(n, name="hsv"):
    return plt.cm.get_cmap(name, n)


loss_name = "Huber"
delta_small = 0.1
delta_large = 10.0
percentage = 0.05
a_values = [0.1, 0.5, 1.0, 2.0, 10.0]

experimental_settings = [
    {
        "loss_name": loss_name,
        "alpha_min": 0.01,
        "alpha_max": 100,
        "alpha_pts": 15,
        "reg_param": 1.5,
        "repetitions": 15,
        "n_features": 500,
        "percentage": percentage,
        "delta_small": delta_small,
        "delta_large": delta_large,
        "a": a,
        "experiment_type": "exp",
    }
    for a in a_values
]

theoretical_settings = [
    {
        "loss_name": loss_name,
        "alpha_min": 0.01,
        "alpha_max": 100,
        "alpha_pts": 35,
        "reg_param": 1.5,
        "n_features": 500,
        "percentage": percentage,
        "delta_small": delta_small,
        "delta_large": delta_large,
        "a": a,
        "experiment_type": "theory",
    }
    for a in a_values
]

n_exp = len(a_values)

alphas_num = [None] * n_exp
errors_mean_num = [None] * n_exp
errors_std_num = [None] * n_exp

alphas_theory = [None] * n_exp
errors_theory = [None] * n_exp

for idx, (exp_dict, theory_dict) in enumerate(
    zip(tqdm(experimental_settings, desc="a values", leave=False), theoretical_settings)
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

pu.initialization_mpl()

fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)

for idx, (al_n, err_m, err_s, al_t, err_t, a) in enumerate(
    zip(
        alphas_num,
        errors_mean_num,
        errors_std_num,
        alphas_theory,
        errors_theory,
        a_values,
    )
):
    colormap = get_cmap(n_exp + 3)
    ax.plot(
        al_t,
        err_t,
        # marker='.',
        color=colormap(idx + 3),
    )

    ax.errorbar(
        al_n,
        err_m,
        err_s,
        marker=".",
        #  linestyle="None",
        color=colormap(idx + 3),
        label=r"$a = {:.1f}$".format(a),
    )

ax.set_title(
    r"{} Loss - $\Delta = [{:.2f}, {:.2f}]$".format(loss_name, delta_small, delta_large)
)
ax.set_ylabel(r"$\frac{1}{d} E[||\hat{w} - w^\star||^2]$")
ax.set_xlabel(r"$\alpha$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([0.009, 110])
ax.legend()

if save:
    pu.save_plot(fig, "{}_together_different_a_{:d}".format(loss_name, random_number))

plt.show()
