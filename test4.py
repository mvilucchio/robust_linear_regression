import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from src.utils import AMP_points_runner, check_saved, bayes_optimal_runner, load_file

n = 100
d = 1000
delta = 0.5
delta_small = 0.5
delta_large = 3.0
deltas_large = [1.0, 3.0, 5.0]
eps = 0.1
# beta = 0.0
n_alpha_points = 15

if __name__ == "__main__":

    AMP_experimental_settings = [
        {
            "alpha_min": 0.5,
            "alpha_max": 100,
            "alpha_pts": 20,
            "repetitions": 10,
            "n_features": d,
            "delta": dl,
            # "delta_small": delta_small,
            # "delta_large": dl,
            # "percentage": eps,
            "experiment_type": "GAMP",
        }
        for dl in deltas_large
    ]

    BO_settings = [
        {
            "alpha_min": 0.01,
            "alpha_max": 100,
            "alpha_pts": 40,
            "delta": dl,
            # "delta_small": delta_small,
            # "delta_large": dl,
            # "percentage": eps,
            "experiment_type": "BO",
        }
        for dl in deltas_large
    ]

    #  AMP_points_runner(**AMP_exp_setting)

    alphas_bo = [None] * len(deltas_large)
    errors_bo = [None] * len(deltas_large)
    alphas_amp = [None] * len(deltas_large)
    errors_mean_amp = [None] * len(deltas_large)
    errors_std_amp = [None] * len(deltas_large)

    for idx, (amp_dict, bo_dict, dl) in enumerate(
        zip(AMP_experimental_settings, BO_settings, deltas_large)
    ):
        print("Doing dl: ", dl)

        file_exists, file_path = check_saved(**bo_dict)

        if not file_exists:
            bayes_optimal_runner(**bo_dict)

        bo_dict.update({"file_path": file_path})
        alphas_bo[idx], errors_bo[idx] = load_file(**bo_dict)
        print("here")
        file_exists, file_path = check_saved(**amp_dict)

        if not file_exists:
            AMP_points_runner(**amp_dict)

        bo_dict.update({"file_path": file_path})
        alphas_amp[idx], errors_mean_amp[idx], errors_std_amp[idx] = load_file(**amp_dict)

    for idx, (a_bo, e_bo, a_amp, em_amp, es_amp) in enumerate(
        zip(alphas_bo, errors_bo, alphas_amp, errors_mean_amp, errors_std_amp)
    ):
        plt.plot(a_bo, e_bo, label=deltas_large[idx])
        plt.errorbar(a_amp, em_amp, es_amp, label=deltas_large[idx])

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()
