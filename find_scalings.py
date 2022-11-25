import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from src.fpeqs import different_alpha_observables_fpeqs
from src.fpeqs_Huber import (
    var_func_L2,
    var_hat_func_Huber_decorrelated_noise,
)

if __name__ == "__main__":

    delta_small = 0.1
    delta_large = 10.0
    percentage = 0.3
    beta = 0.0
    a = 1.0
    var_hat_kwargs = {
        "delta_small": delta_small,
        "delta_large": delta_large,
        "percentage": percentage,
        "beta": beta,
        "a": a,
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
        var_hat_func_Huber_decorrelated_noise,
        funs=[lambda m, q, sigma: m, lambda m, q, sigma: q, lambda m, q, sigma: sigma],
        alpha_1=0.01,
        alpha_2=100000,
        n_alpha_points=100,
        reg_param=0.1,
        initial_cond=initial_condition,
        var_hat_kwargs=var_hat_kwargs,
    )

    small_sqrt = delta_small - 2 * m + q + 1
    large_sqrt = delta_large - 2 * m * beta + q + beta**2
    small_erf = (a * (sigma + 1)) / np.sqrt(2 * small_sqrt)
    large_erf = (a * (sigma + 1)) / np.sqrt(2 * large_sqrt)

    mhat = (alphas / (1 + sigma)) * (
        (1 - percentage) * erf(small_erf) + beta * percentage * erf(large_erf)
    )
    qhat = alphas * (
        a**2
        - (np.sqrt(2 / np.pi) * a / (1 + sigma))
        * (
            (1 - percentage) * np.sqrt(small_sqrt) * np.exp(-(small_erf**2))
            + percentage * np.sqrt(large_sqrt) * np.exp(-(large_erf**2))
        )
        + (1 / (1 + sigma) ** 2)
        * (
            (1 - percentage) * (small_sqrt - (a * (1 + sigma)) ** 2) * erf(small_erf)
            + percentage * (large_sqrt - (a * (1 + sigma)) ** 2) * erf(large_erf)
        )
    )
    sigmahat = (alphas / (1 + sigma)) * (
        (1 - percentage) * erf(small_erf) + percentage * erf(large_erf)
    )

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
