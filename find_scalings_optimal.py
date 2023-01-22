import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from src.fpeqs import different_alpha_observables_fpeqs, _find_fixed_point
from src.fpeqs_Huber import (
    var_func_L2,
    var_hat_func_Huber_decorrelated_noise,
)
from src.optimal_lambda import (
    optimal_lambda,
    optimal_reg_param_and_huber_parameter,
    no_parallel_optimal_reg_param_and_huber_parameter,
)

if __name__ == "__main__":

    delta_small = 0.1
    delta_large = 10.0
    percentage = 0.3
    beta = 0.3
    # a = 1.0
    # var_hat_kwargs = {
    #     "delta_small": delta_small,
    #     "delta_large": delta_large,
    #     "percentage": percentage,
    #     "beta": beta,
    #     "a": a,
    # }
    var_hat_kwargs = {
        "delta_small": delta_small,
        "delta_large": delta_large,
        "percentage": percentage,
        "beta": beta,
    }

    while True:
        m = 0.89 * np.random.random() + 0.1
        q = 0.89 * np.random.random() + 0.1
        sigma = 0.89 * np.random.random() + 0.1
        if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
            initial_condition = [m, q, sigma]
            break

    (alphas, errors, lambdas, huber_params,) = no_parallel_optimal_reg_param_and_huber_parameter(
        var_hat_func=var_hat_func_Huber_decorrelated_noise,
        alpha_1=10000,
        alpha_2=10000000000,
        n_alpha_points=150,
        initial_cond=initial_condition,
        var_hat_kwargs=var_hat_kwargs,
    )
    print(alphas, lambdas, huber_params)

    lll = len(lambdas)

    ms = np.zeros(lll)
    qs = np.zeros(lll)
    sigmas = np.zeros(lll)

    mhats = np.zeros(lll)
    qhats = np.zeros(lll)
    sigmahats = np.zeros(lll)

    for idx, (alph, l, a) in enumerate(zip(alphas, lambdas, huber_params)):
        ms[idx], qs[idx], sigmas[idx] = _find_fixed_point(alph, var_func_L2, var_hat_func_Huber_decorrelated_noise, l, initial_condition, var_hat_kwargs)
        # print(ms[idx], qs[idx], sigmas[idx], delta_small, delta_large, beta, "^^^", a, l)

        small_sqrt = delta_small - 2 * ms[idx] + qs[idx] + 1
        large_sqrt = delta_large - 2 * ms[idx] * beta + qs[idx] + beta**2
        small_erf = (a * (sigmas[idx] + 1)) / np.sqrt(2 * small_sqrt)
        large_erf = (a * (sigmas[idx] + 1)) / np.sqrt(2 * large_sqrt)

        mhats[idx] = (alph / (1 + sigmas[idx])) * (
            (1 - percentage) * erf(small_erf) + beta * percentage * erf(large_erf)
        )
        qhats[idx] = alph * (
            a**2
            - (np.sqrt(2 / np.pi) * a / (1 + sigmas[idx]))
            * (
                (1 - percentage) * np.sqrt(small_sqrt) * np.exp(-(small_erf**2))
                + percentage * np.sqrt(large_sqrt) * np.exp(-(large_erf**2))
            )
            + (1 / (1 + sigmas[idx]) ** 2)
            * (
                (1 - percentage) * (small_sqrt - (a * (1 + sigmas[idx])) ** 2) * erf(small_erf)
                + percentage * (large_sqrt - (a * (1 + sigmas[idx])) ** 2) * erf(large_erf)
            )
        )
        sigmahats[idx] = (alph / (1 + sigmas[idx])) * (
            (1 - percentage) * erf(small_erf) + percentage * erf(large_erf)
        )

    # estimate the exponent
    def log_difference(m1,m2,a1,a2):
        return (np.log10(m1) - np.log10(m2)) / (np.log10(a1) - np.log10(a2))

    print("ms ", log_difference(ms[-1],ms[-10],alphas[-1],alphas[-10]))
    print("qs ", log_difference(qs[-1],qs[-10],alphas[-1],alphas[-10]))
    print("sigmas ", log_difference(sigmas[-1],sigmas[-10],alphas[-1],alphas[-10]))
    print("mhats ", log_difference(mhats[-1],mhats[-10],alphas[-1],alphas[-10]))
    print("qhats ", log_difference(qhats[-1],qhats[-10],alphas[-1],alphas[-10]))
    print("sigmahats ", log_difference(sigmahats[-1],sigmahats[-10],alphas[-1],alphas[-10]))
    print("huber_param ", log_difference(huber_params[-1],huber_params[-10],alphas[-1],alphas[-10]))

    plt.plot(alphas, ms, label="m")
    plt.plot(alphas, qs, label="qs[idx]")
    plt.plot(alphas, sigmas, label="sigma")
    plt.plot(alphas, mhats, label="mhat")
    plt.plot(alphas, qhats, label="qhat")
    plt.plot(alphas, sigmahats, label="sigmahat")
    plt.plot(alphas, huber_params, label="huber_parrams")
    plt.title("Delta_small {} Delta_large {} eps {} beta {}".format(delta_small, delta_large, percentage, beta))
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(which='both')
    plt.legend()

    plt.show()
