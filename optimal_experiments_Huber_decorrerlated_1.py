from src.utils import experiment_runner, load_file
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from src.fpeqs import no_parallel_different_alpha_observables_fpeqs_parallel
from src.fpeqs_Huber import var_func_L2, var_hat_func_Huber_decorrelated_noise

if __name__ == "__main__":
    percentage, delta_small, delta_large = 0.3, 0.1, 5.0

    deltas_large = [0.5, 1.0, 2.0, 5.0, 10.0]
    percentages = [0.1, 0.3]
    betas = [0.0]
    dl = 10.0
    b = betas[0]

    experiments_settings = [
        {
            "loss_name": "Huber",
            "alpha_min": 1000,
            "alpha_max": 10000000,
            "alpha_pts": 200,
            "delta_small": delta_small,
            "delta_large": dl,
            "percentage": percentage,
            "beta": b,
            "experiment_type": "reg_param huber_param optimal",
        }
        #  for dl in deltas_large  #  for p in percentages  #
    ]

    for exp_dict in tqdm(experiments_settings):
        experiment_runner(**exp_dict)

    _, _, reg_param, huberspar = load_file(
        **{
            "loss_name": "Huber",
            "alpha_min": 1000,
            "alpha_max": 10000000,
            "alpha_pts": 200,
            "delta_small": delta_small,
            "delta_large": dl,
            "percentage": percentage,
            "beta": b,
            "experiment_type": "reg_param huber_param optimal",
        }
    )

    while True:
        m = 0.89 * np.random.random() + 0.1
        q = 0.89 * np.random.random() + 0.1
        sigma = 0.89 * np.random.random() + 0.1
        if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
            initial_condition = [m, q, sigma]
            break

    alphas, [m, q, sigma] = no_parallel_different_alpha_observables_fpeqs_parallel(
        var_func_L2,
        var_hat_func_Huber_decorrelated_noise,
        funs=[lambda m, q, sigma: m, lambda m, q, sigma: q, lambda m, q, sigma: sigma],
        alpha_1=1000,
        alpha_2=10000000,
        n_alpha_points=200,
        reg_param=reg_param,
        initial_cond=initial_condition,
        var_hat_kwargs=[
            {
                "delta_small": delta_small,
                "delta_large": delta_large,
                "percentage": percentage,
                "beta": b,
                "a": a,
            }
            for a in huberspar
        ],
    )

    small_sqrt = delta_small - 2 * m + q + 1
    large_sqrt = delta_large - 2 * m * b + q + b ** 2
    small_erf = (huberspar * (sigma + 1)) / np.sqrt(2 * small_sqrt)
    large_erf = (huberspar * (sigma + 1)) / np.sqrt(2 * large_sqrt)

    mhat = (alphas / (1 + sigma)) * (
        (1 - percentage) * erf(small_erf) + b * percentage * erf(large_erf)
    )
    qhat = alphas * (
        huberspar ** 2
        - (np.sqrt(2 / np.pi) * huberspar / (1 + sigma))
        * (
            (1 - percentage) * np.sqrt(small_sqrt) * np.exp(-(small_erf ** 2))
            + percentage * np.sqrt(large_sqrt) * np.exp(-(large_erf ** 2))
        )
        + (1 / (1 + sigma) ** 2)
        * (
            (1 - percentage)
            * (small_sqrt - (huberspar * (1 + sigma)) ** 2)
            * erf(small_erf)
            + percentage * (large_sqrt - (huberspar * (1 + sigma)) ** 2) * erf(large_erf)
        )
    )
    sigmahat = (alphas / (1 + sigma)) * (
        (1 - percentage) * erf(small_erf) + percentage * erf(large_erf)
    )

    #  plt.plot(alphas, huberspar)
    plt.plot(alphas, m, label="m")
    plt.plot(alphas, q, label="q")
    plt.plot(alphas, sigma, label="sigma")
    plt.plot(alphas, mhat, label="mhat")
    plt.plot(alphas, qhat, label="qhat")
    plt.plot(alphas, sigmahat, label="sigmahat")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(which="both", axis="both")
    plt.show()
