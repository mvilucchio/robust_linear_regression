import numpy as np
import matplotlib.pyplot as plt
from src.utils import experiment_runner, load_file
from tqdm.auto import tqdm
import src.fpeqs as fpe
from src.fpeqs_BO import var_func_BO, var_hat_func_BO_num_double_noise

if __name__ == "__main__":
    random_number = np.random.randint(0, 100)

    names_cm = ["Purples", "Blues", "Greens", "Oranges", "Greys"]

    def get_cmap(n, name="hsv"):
        return plt.cm.get_cmap(name, n)

    delta_small, delta_large, percentage = 0.1, 10.0, 0.3

    while True:
        m = 0.89 * np.random.random() + 0.1
        q = 0.89 * np.random.random() + 0.1
        sigma = 0.89 * np.random.random() + 0.1
        if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
            initial_condition = [m, q, sigma]
            break

    deltas_large = [0.5, 2.0, 5.0, 10.0]
    experiments_settings = [
        {
            "alpha_min": 0.01,
            "alpha_max": 100,
            "alpha_pts": 32,
            "delta_small": delta_small,
            "delta_large": delta_large,
            "percentage": percentage,
            # "beta": b,
            "experiment_type": "BO",
        },
        {
            "alpha_min": 0.01,
            "alpha_max": 100,
            "alpha_pts": 32,
            "delta_small": delta_large,
            "delta_large": delta_small,
            "percentage": 1 - percentage,
            # "beta": b,
            "experiment_type": "BO",
        },
    ]

    # alphas_single, (errors_single,) = fpe.different_alpha_observables_fpeqs(
    #     fpe.var_func_BO,
    #     fpe.var_hat_func_BO_single_noise,
    #     alpha_1=0.01,
    #     alpha_2=1000,
    #     n_alpha_points=32,
    #     initial_cond=initial_condition,
    #     var_hat_kwargs={"delta": delta_small},
    # )

    alpha = []
    errors = []

    ls = [0.01, 0.05, 0.1, 0.3]
    for p in tqdm(ls):
        alphas_double, (errors_double,) = fpe.different_alpha_observables_fpeqs(
            var_func_BO,
            var_hat_func_BO_num_double_noise,
            alpha_1=0.01,
            alpha_2=1000,
            n_alpha_points=32,
            initial_cond=initial_condition,
            var_hat_kwargs={
                "delta_small": delta_small,
                "delta_large": delta_large,
                "percentage": p,
            },
        )
        alpha.append(alphas_double)
        errors.append(errors_double)

    for a, e, p in zip(alpha, errors, ls):
        plt.plot(a, e, label="percentage {}".format(p))
    # print(errors_single - errors_double)
    # print(errors_single - errors_flipped)
    # print(np.abs(errors_double - errors_flipped))
    # print(np.abs(errors_double_less - errors_flipped_less))

    # for idx, exp_d in enumerate(tqdm(experiments_settings)):
    #     experiment_runner(**exp_d)

    # for idx, exp_d in enumerate(tqdm(experiments_settings)):
    #     alphas_t, err_t = load_file(**exp_d)
    #     plt.plot(
    #         alphas_t,
    #         err_t,
    #         label="ds = {} dl = {}".format(exp_d["delta_small"], exp_d["delta_large"]),
    #     )

    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("gen. error")
    plt.xlabel(r"$\alpha$")
    plt.legend()
    plt.grid()
    plt.show()
