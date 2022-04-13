import matplotlib.pyplot as plt
from src.utils import check_saved, load_file

alpha_min, alpha_max, alpha_points = 0.01, 100, 36
delta_small, delta_large = 0.1, 10.0
e = 0.3

experiments = [
    {
        "loss_name": "L2",
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "alpha_pts": alpha_points,
        "delta_small": delta_small,
        "delta_large": delta_large,
        "epsilon": e,
        "experiment_type": "reg param optimal",
    },
    {
        "loss_name": "Huber",
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "alpha_pts": alpha_points,
        "delta_small": delta_small,
        "delta_large": delta_large,
        "epsilon": e,
        "experiment_type": "reg param optimal",
    },
    {
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "alpha_pts": alpha_points,
        "delta_small": delta_small,
        "delta_large": delta_large,
        "epsilon": e,
        "experiment_type": "BO",
    },
]

alphas = []
errors = []

fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)

for exper in experiments:
    file_exists, file_path = check_saved(**exper)

    if not file_exists:
        raise RuntimeError("File doesn't exists")
    else:
        exper.update({"file_path": file_path})

        if exper.get("experiment_type") == "reg param optimal":
            a, e, _ = load_file(**exper)
            alphas.append(a)
            errors.append(e)
        else:
            a, e = load_file(**exper)
            alphas.append(a)
            errors.append(e)


for a, e, exp in zip(alphas, errors, experiments):
    ax.plot(
        a,
        e,
        label=r"$\epsilon$ = {epsilon} $\Delta$ = [{delta_small}, {delta_large}]".format(
            **exp
        ),
    )

ax.set_ylabel(r"$\frac{1}{d} E[||\hat{w} - w^\star||^2]$")
ax.set_xlabel(r"$\alpha$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.minorticks_on()
ax.grid(True, which="both")
ax.legend()

plt.show()
