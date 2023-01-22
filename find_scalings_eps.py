import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
import src.fpeqs as fp
from scipy.optimize import minimize
from src.fpeqs import different_alpha_observables_fpeqs
from src.fpeqs_Huber import (
    var_func_L2,
    var_hat_func_Huber_decorrelated_noise,
)
from src.fpeqs_L1 import (
    # var_func_L2,
    var_hat_func_L1_decorrelated_noise,
)
from src.fpeqs_L2 import (
    # var_func_L2,
    var_hat_func_L2_decorrelated_noise,
)

SMALLEST_REG_PARAM = 1e-8
SMALLEST_HUBER_PARAM = 1e-8
MAX_ITER = 2500
XATOL = 1e-9
FATOL = 1e-9

def _find_optimal_reg_param_and_huber_parameter_gen_error(
    alpha, var_hat_func, initial, var_hat_kwargs, inital_values
):
    def minimize_fun(x):
        reg_param, a = x
        var_hat_kwargs.update({"a": a})
        m, q, _ = fp.state_equations(
            var_func_L2,
            var_hat_func,
            reg_param=reg_param,
            alpha=alpha,
            init=initial,
            var_hat_kwargs=var_hat_kwargs,
        )
        return 1 + q - 2 * m

    bnds = [(SMALLEST_REG_PARAM, None), (SMALLEST_HUBER_PARAM, None)]
    obj = minimize(
        minimize_fun,
        x0=inital_values,
        method="Nelder-Mead",
        bounds=bnds,
        options={
            "xatol": XATOL,
            "fatol": FATOL,
            "adaptive": True,
        },
    )

    if obj.success:
        fun_val = obj.fun
        reg_param_opt, a_opt = obj.x
        var_hat_kwargs.update({"a": a_opt})

        m, q, sigma = fp.state_equations(
            var_func_L2,
            var_hat_func,
            reg_param=reg_param_opt,
            alpha=alpha,
            init=initial,
            var_hat_kwargs=var_hat_kwargs,
        )
        percentage = var_hat_kwargs.get("percentage")

        small_sqrt = delta_small - 2 * m + q + 1
        large_sqrt = delta_large - 2 * m * beta + q + beta**2
        small_erf = (a_opt * (sigma + 1)) / np.sqrt(2 * small_sqrt)
        large_erf = (a_opt * (sigma + 1)) / np.sqrt(2 * large_sqrt)

        mhat = (alpha / (1 + sigma)) * (
            (1 - percentage) * erf(small_erf) + beta * percentage * erf(large_erf)
        )
        qhat = alpha * (
            a_opt**2
            - (np.sqrt(2 / np.pi) * a_opt / (1 + sigma))
            * (
                (1 - percentage) * np.sqrt(small_sqrt) * np.exp(-(small_erf**2))
                + percentage * np.sqrt(large_sqrt) * np.exp(-(large_erf**2))
            )
            + (1 / (1 + sigma) ** 2)
            * (
                (1 - percentage) * (small_sqrt - (a_opt * (1 + sigma)) ** 2) * erf(small_erf)
                + percentage * (large_sqrt - (a_opt * (1 + sigma)) ** 2) * erf(large_erf)
            )
        )
        sigmahat = (alpha / (1 + sigma)) * (
            (1 - percentage) * erf(small_erf) + percentage * erf(large_erf)
        )
        return m, q, sigma, mhat, qhat, sigmahat, reg_param_opt, a_opt
    else:
        raise RuntimeError("Minima could not be found.")

if __name__ == "__main__":

    delta_small = 0.1
    delta_large = 10.0
    beta = 0.0
    a = 1

    while True:
        m = 0.89 * np.random.random() + 0.1
        q = 0.89 * np.random.random() + 0.1
        sigma = 0.89 * np.random.random() + 0.1
        if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
            initial_condition = [m, q, sigma]
            break

    eps_zero = 0.0
    epsilons = np.logspace(-7,-2, 40)
    m = np.empty_like(epsilons)
    q = np.empty_like(epsilons)
    sigma = np.empty_like(epsilons)
    mhat = np.empty_like(epsilons)
    qhat = np.empty_like(epsilons)
    sigmahat = np.empty_like(epsilons)
    a_hub = np.empty_like(epsilons)

    for idx, eps in enumerate(epsilons):
        print(idx)
        params = {
            "delta_small": delta_small,
            "delta_large": delta_large,
            "percentage": float(eps),
            "beta": beta,
            "a": 10
        }

        m[idx], q[idx], sigma[idx], mhat[idx], qhat[idx], sigmahat[idx], _, a_hub[idx]  = _find_optimal_reg_param_and_huber_parameter_gen_error(
            10,
            var_hat_func_Huber_decorrelated_noise,
            initial_condition,
            params,
            [0.5, 10],
        )

    print("begin")
    params = {
        "delta_small": delta_small,
        "delta_large": delta_large,
        "percentage": float(eps_zero),
        "beta": beta,
        "a": 10
    }

    m_zero, q_zero, sigma_zero, mhat_zero, qhat_zero, sigmahat_zero, _,_  = _find_optimal_reg_param_and_huber_parameter_gen_error(
        10,
        var_hat_func_Huber_decorrelated_noise,
        initial_condition,
        params,
        [0.5, 10],
    )
    print("done")

    # epsilons = epsilons * np.log(1/epsilons)

    mhat1 = (mhat- mhat_zero)/mhat_zero
    sigma1 = (sigma- sigma_zero)/sigma_zero
    mhat0 = mhat_zero
    sigma0 = sigma_zero
    q0 = (q - q_zero)/q_zero
    m0 = (m - m_zero)/m_zero

    # plt.plot(epsilons, mhat1 *(1+sigma0) + sigma1 * mhat0 )
    g0 = (1+sigma0)/np.sqrt(2*(delta_small + 1 + q0 - 2 * m0))
    plt.plot(a_hub , np.sqrt(-np.log(epsilons) ))

    # plt.plot(epsilons, np.abs(m - m_zero)/m_zero, label="m")
    # plt.plot(epsilons, np.abs(q - q_zero)/q_zero, label="q")
    # plt.plot(epsilons, np.abs(sigma- sigma_zero)/sigma_zero, label="sigma")
    # plt.plot(epsilons, np.abs(mhat- mhat_zero)/mhat_zero, label="mhat")
    # plt.plot(epsilons, np.abs(qhat - qhat_zero)/qhat_zero, label="qhat")
    # plt.plot(epsilons, np.abs(sigmahat - sigmahat_zero)/sigmahat_zero, label="sigmahat")
    # plt.plot(epsilons, np.abs((sigma+1)/np.sqrt(2*(delta_small + 1 +q -2*m)) -(sigma_zero+1)/np.sqrt(2*(delta_small + 1 +q_zero -2*m_zero))) , label="aaa")

    # plt.xscale("log")
    # plt.yscale("log")
    # plt.ylim([0.5, 1.5])
    plt.grid(which='both')
    plt.legend()
    # plt.annotate("{:.3f}".format(m[-1]), (10**4, m[-1]))
    # plt.annotate("{:.3f}".format(q[-1]), (10**4, q[-1]))

    # print(1-epsilons, m[-1],q[-1], sigma[-1])

    plt.show()
