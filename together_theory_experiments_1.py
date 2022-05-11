import numpy as np
import matplotlib.pyplot as plt
import src.plotting_utils as pu
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

    loss_name = "Huber"
    delta_small, delta_large, percentage, beta = 0.1, 5.0, 0.1, 0.0
    reg_params = [0.01, 0.1, 1.0, 10.0, 100.0]

    experimental_settings = [
        {
            "loss_name": loss_name,
            "alpha_min": 0.01,
            "alpha_max": 2000,
            "alpha_pts": 20,
            "reg_param": reg_param,
            "repetitions": 10,
            "n_features": 600,
            # "delta" : delta_large,
            "delta_small": delta_small,
            "delta_large": delta_large,
            "percentage": percentage,
            "beta": beta,
            "a": 1.0,
            "experiment_type": "exp",
        }
        for reg_param in reg_params
    ]

    theory_settings = [
        {
            "loss_name": loss_name,
            "alpha_min": 0.01,
            "alpha_max": 1000,
            "alpha_pts": 100,
            "reg_param": reg_param,
            # "delta" : delta_large,
            "delta_small": delta_small,
            "delta_large": delta_large,
            "percentage": percentage,
            "beta" : beta,
            "a": 1.0,
            "experiment_type": "theory",
        }
        for reg_param in reg_params
    ]

    n_exp = len(theory_settings)

    for idx, (exp_dict, theory_dict) in enumerate(
        zip(experimental_settings, theory_settings)
    ):
        experiment_runner(**theory_dict)

        experiment_runner(**exp_dict)

