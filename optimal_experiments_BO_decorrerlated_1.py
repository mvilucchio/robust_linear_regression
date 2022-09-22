from src.utils import experiment_runner, load_file
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import src.fpeqs as fpe
from src.fpeqs_BO import var_func_BO, var_hat_func_BO_num_double_noise


if __name__ == "__main__":
    percentage, delta_small, delta_large = 0.1, 0.1, 5.0

    deltas_large = [0.5, 1.0, 5.0, 10.0]
    betas = [0.0]
    b = betas[0]
    percentages = [0.05, 0.1, 0.3]
    # dl = 5.0
    p = 0.1

    experiments_settings = [
        {
            # "loss_name": "Huber",
            "alpha_min": 0.01,
            "alpha_max": 100,
            "alpha_pts": 30,
            "delta_small": delta_small,
            "delta_large": dl,
            "percentage": p,
            # "beta": b,
            "experiment_type": "BO",
        }
        for dl in deltas_large  # for p in percentages  # for dl in deltas_large  #    # for dl in deltas_large
    ]

    alpha = []
    errors = []

    ls = [0.5, 1.0, 2.0, 5.0, 10.0]
    for dl in tqdm(ls):

        while True:
            m = 0.89 * np.random.random() + 0.1
            q = 0.89 * np.random.random() + 0.1
            sigma = 0.89 * np.random.random() + 0.1
            if np.square(m) < q + delta_small * q and np.square(m) < q + dl * q:
                initial_condition = [m, q, sigma]
                break

        alphas_double, (errors_double,) = fpe.different_alpha_observables_fpeqs(
            var_func_BO,
            var_hat_func_BO_num_double_noise,
            alpha_1=0.01,
            alpha_2=100,
            n_alpha_points=32,
            initial_cond=initial_condition,
            var_hat_kwargs={
                "delta_small": delta_small,
                "delta_large": dl,
                "percentage": p,
            },
        )
        alpha.append(alphas_double)
        errors.append(errors_double)

    # for exp_dict in tqdm(experiments_settings):
    #     experiment_runner(**exp_dict)

    file_names = ["./BOGE epsilon 0.1 delta1 0.1 delta2 {}.npz".format(dl) for dl in ls]

    for a, e, p in zip(alpha, errors, ls):
        plt.plot(a, e, label="DL {}".format(p))

    for dl, fn, exp_dict in zip(deltas_large, file_names, experiments_settings):
        dat = np.load(fn, mmap_mode="r")
        al = dat["alphas"]
        err = dat["errors"]

        plt.scatter(al, err, label="AMP DL {}".format(dl))

        # a, e = load_file(**exp_dict)

        # plt.plot(a, e, label="BO DL {}".format(dl))

    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("gen. error")
    plt.xlabel(r"$\alpha$")
    plt.legend()
    plt.grid()
    plt.show()
