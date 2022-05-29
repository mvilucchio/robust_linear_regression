import matplotlib.pyplot as plt
import numpy as np
from src.fpeqs import (
    different_alpha_observables_fpeqs,
    var_func_L2,
    var_hat_func_L2_decorrelated_noise,
    var_hat_func_L2_double_noise,
    var_hat_func_Huber_num_decorrelated_noise,
)

if __name__ == "__main__":

    delta_small = 0.1
    delta_large = 10.0
    percentage = 0.1
    beta = 0.5

    var_hat_kwargs = {
        "delta_small": delta_small,
        "delta_large": delta_large,
        "percentage": percentage,
        "beta": beta,
        "a": 1.0,
    }

    while True:
        m = 0.89 * np.random.random() + 0.1
        q = 0.89 * np.random.random() + 0.1
        sigma = 0.89 * np.random.random() + 0.1
        if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
            initial_condition = [m, q, sigma]
            break

    alphas, [m, q, sigma] = different_alpha_observables_fpeqs(
        var_func_L2,
        var_hat_func_Huber_num_decorrelated_noise,
        funs=[lambda m, q, sigma: m, lambda m, q, sigma: q, lambda m, q, sigma: sigma],
        alpha_1=0.01,
        alpha_2=100000,
        n_alpha_points=100,
        reg_param=0.1,
        initial_cond=initial_condition,
        var_hat_kwargs=var_hat_kwargs,
    )

    delta_eff = (1 - percentage) * delta_small + percentage * delta_large
    intermediate_val = 1 + percentage * (beta - 1)
    mhat = alphas / (1 + sigma) * (1 + percentage * (beta - 1))
    qhat = (
        alphas
        * (
            1
            + q
            + delta_eff
            + percentage * (beta ** 2 - 1)
            - 2 * np.abs(m) * intermediate_val
        )
        / ((1 + sigma) ** 2)
    )
    sigmahat = alphas / (1 + sigma)

    plt.plot(alphas, m, label="m")
    plt.plot(alphas, q, label="q")
    plt.plot(alphas, sigma, label="sigma")
    plt.plot(alphas, mhat, label="mhat")
    plt.plot(alphas, qhat, label="qhat")
    plt.plot(alphas, sigmahat, label="sigmahat")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.legend()

    plt.show()
