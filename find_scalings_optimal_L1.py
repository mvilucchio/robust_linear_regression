import matplotlib.pyplot as plt
import numpy as np
import src.plotting_utils as pu
# from scipy.special import erf
from math import erf, erfc, exp, log, sqrt
from src.fpeqs import different_alpha_observables_fpeqs, _find_fixed_point
from src.fpeqs_Huber import (
    var_func_L2,
    var_hat_func_Huber_decorrelated_noise,
)
from src.fpeqs_L1 import (
    var_hat_func_L1_decorrelated_noise,
)
from src.optimal_lambda import (
    optimal_lambda,
    no_parallel_optimal_lambda,
    optimal_reg_param_and_huber_parameter,
    no_parallel_optimal_reg_param_and_huber_parameter,
)

if __name__ == "__main__":

    save = True
    width = 1.0 * 458.63788

    delta_small = 1.0
    beta = 0.0
    delta_large = 10.0
    percentage = 0.3

    UPPER_BOUND = 1e10
    LOWER_BOUND = 1e6

    pu.initialization_mpl()

    tuple_size = pu.set_size(width, fraction=0.50)

    fig, ax = plt.subplots(1, 1, figsize=tuple_size)
    fig.subplots_adjust(left=0.16)
    fig.subplots_adjust(bottom=0.16)
    fig.subplots_adjust(top=0.97)
    fig.subplots_adjust(right=0.97)
    

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
    
    # changeee  
    (alphas, errors, lambdas) = no_parallel_optimal_lambda(
        var_func_L2,
        var_hat_func_L1_decorrelated_noise,
        alpha_1=LOWER_BOUND,
        alpha_2=UPPER_BOUND,
        n_alpha_points=500,
        initial_cond=initial_condition,
        var_hat_kwargs=var_hat_kwargs,
    )
    print(alphas, lambdas)

    lll = len(lambdas)

    ms = np.zeros(lll)
    qs = np.zeros(lll)
    sigmas = np.zeros(lll)

    mhats = np.zeros(lll)
    qhats = np.zeros(lll)
    sigmahats = np.zeros(lll)

    for idx, (alph, l) in enumerate(zip(alphas, lambdas)):
        ms[idx], qs[idx], sigmas[idx] = _find_fixed_point(alph, var_func_L2, var_hat_func_L1_decorrelated_noise, l, initial_condition, var_hat_kwargs)

        mhats[idx], qhats[idx], sigmahats[idx] =  var_hat_func_L1_decorrelated_noise(m, q, sigma, alph, delta_small, delta_large, percentage, beta)

    ax.plot(alphas, ms, label=r"$m$")
    ax.plot(alphas, qs, label=r"$q$")
    ax.plot(alphas, sigmas, label=r"$\Sigma$")
    ax.plot(alphas, mhats, label=r"$\hat{m}$")
    ax.plot(alphas, qhats, label=r"$\hat{q}$")
    ax.plot(alphas, sigmahats, label=r"$\hat{\Sigma}$")

    ax.set_xlabel(r"$\alpha$", labelpad=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([LOWER_BOUND, UPPER_BOUND])
    # ax_2.set_ylim([1e-7, 1.5])
    ax.legend(ncol=2, handlelength=1.0)

    ax.tick_params(axis="y", pad=2.0)
    ax.tick_params(axis="x", pad=2.0)

    # ax.set_xticks([0.0001, 0.001, 0.01, 0.1, 0.5])
    # ax_2.set_xticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$0.5$"])

    if save:
        pu.save_plot(
            fig,
            "large_alpha_scalinigs_delta_large_{:.2f}_beta_{:.2f}_delta_small_{:.2f}_eps_{:.2f}".format(  # "a_hub_sweep_eps_fixed_delta_{:.2f}_beta_{:.2f}_alpha_cut_{:.2f}".format( # "sweep_eps_fixed_delta_{:.2f}_beta_{:.2f}_alpha_cut_{:.2f}".format(
                delta_large, beta, delta_small, percentage
            ),
        )

    plt.show()

    plt.show()
