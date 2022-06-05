import numpy as np
import matplotlib.pyplot as plt
from src.utils import experiment_runner, load_file
from tqdm.auto import tqdm

if __name__ == "__main__":
    random_number = np.random.randint(0, 100)

    names_cm = ["Purples", "Blues", "Greens", "Oranges", "Greys"]

    def get_cmap(n, name="hsv"):
        return plt.cm.get_cmap(name, n)

    loss_name = "Huber"
    delta_small, delta_large, percentage, beta = 0.1, 10.0, 0.3, 0.0
    reg_params = [1e-5]
    # 1e-7, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0

    experimental_settings = [
        {
            "loss_name": loss_name,
            "alpha_min": 0.01,
            "alpha_max": 100,
            "alpha_pts": 7,
            "reg_param": reg_param,
            "repetitions": 5,
            "n_features": 200,
            # "delta": delta_large,
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
            "alpha_max": 10000,
            "alpha_pts": 1000,
            "reg_param": reg_param,
            #  "delta": delta_large,
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

    for idx, (exp_dict, theory_dict) in enumerate(
        zip(tqdm(experimental_settings), theory_settings)
    ):
        experiment_runner(**theory_dict)

        # experiment_runner(**exp_dict)

    for idx, (exp_dict, theory_dict, rp) in enumerate(
        zip(tqdm(experimental_settings), theory_settings, reg_params)
    ):
        alphas_t, err_t = load_file(**theory_dict)
        # alphas_e, err_e, std_e = load_file(**exp_dict)

        plt.plot(alphas_t, err_t, label="{}".format(rp))
        # plt.errorbar(
        #     alphas_e, err_e, std_e, label="{}".format(rp), linestyle="None", marker="."
        # )

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()
