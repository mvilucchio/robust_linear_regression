import numpy as np
import matplotlib.pyplot as plt
import src.fpeqs as fpe
import src.numerics as num
from tqdm.auto import tqdm
from itertools import product
from src.utils import check_saved, load_file, save_file, experiment_runner

random_number = np.random.randint(0, 100)

names_cm = ["Purples", "Blues", "Greens", "Oranges", "Greys"]


def get_cmap(n, name="hsv"):
    return plt.cm.get_cmap(name, n)


loss_name = "Huber"
delta_small = 0.1
deltas_large = [2.0, 5.0, 10.0]
percentages = [0.05, 0.3]

experiments_settings = [
    {
        "alpha_min": 0.01,
        "alpha_max": 100,
        "alpha_pts": 5,
        "n_features": 500,
        "percentage": p,
        "delta_small": delta_small,
        "delta_large": dl,
        "experiment_type": "reg_param huber_param optimal",
    }
    for (dl, p) in product(deltas_large, percentages)
]

n_exp = len(deltas_large) * len(percentages)

alphas = [None] * n_exp
errors = [None] * n_exp
reg_params_opt = [None] * n_exp
huber_params_opt = [None] * n_exp

for idx, exp_dict in enumerate(
    tqdm(experiments_settings, desc="experiments", leave=False)
):
    file_exists, file_path = check_saved(**exp_dict)

    if not file_exists:
        experiment_runner(**exp_dict)

    exp_dict.update({"file_path": file_path})
    alphas[idx], errors[idx], reg_params_opt[idx], huber_params_opt[idx] = load_file(
        **exp_dict
    )


# ------------

fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)

for idx, dl in enumerate(deltas_large):
    colormap = get_cmap(n_exp + 3, name=names_cm[idx])

    for jdx, p in enumerate(percentages):
        i = idx * len(percentages) + jdx

        ax.plot(
            alphas[i],
            errors[i],
            # marker='.',
            label=r"$\Delta_\ell = {:.2f}$ $\epsilon = {:.2f}$".format(dl, p),
            color=colormap(idx + 3),
        )

ax.set_title(r"{} Loss - $\Delta_s = {:.2f}$".format(loss_name, delta_small))
ax.set_ylabel(r"$\frac{1}{d} E[||\hat{w} - w^\star||^2]$")
ax.set_xlabel(r"$\alpha$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([0.009, 110])
ax.minorticks_on()
ax.grid(True, which="both")
ax.legend()

fig.savefig(
    "./imgs/Huber double optim - {}.png".format(random_number), format="png", dpi=150,
)

plt.show()
