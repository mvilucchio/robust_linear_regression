import numpy as np

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import src.plotting_utils as pu

from scipy.optimize import minimize, curve_fit
import src.fpeqs as fp
from src.fpeqs_BO import (
    var_func_BO,
    var_hat_func_BO_single_noise,
    var_hat_func_BO_num_double_noise,
    var_hat_func_BO_num_decorrelated_noise,
)
from src.fpeqs_L2 import (
    var_func_L2,
    var_hat_func_L2_single_noise,
    var_hat_func_L2_double_noise,
    var_hat_func_L2_decorrelated_noise,
)
from src.fpeqs_L1 import (
    var_hat_func_L1_single_noise,
    var_hat_func_L1_double_noise,
    var_hat_func_L1_decorrelated_noise,
)
from src.fpeqs_Huber import (
    var_hat_func_Huber_single_noise,
    var_hat_func_Huber_double_noise,
    var_hat_func_Huber_decorrelated_noise,
)

SMALLEST_REG_PARAM = 1e-8
SMALLEST_HUBER_PARAM = 1e-8
MAX_ITER = 2500
XATOL = 1e-9
FATOL = 1e-9

save = True
experimental_points = True
width = 1.0 * 458.63788
# width = 398.3386
random_number = np.random.randint(100)

alpha_cut = 10.0
delta_small = 0.1
delta_large = 5.0
beta = 0.0

pu.initialization_mpl()

tuple_size = pu.set_size(width, fraction=0.49)

fig, ax = plt.subplots(1, 1, figsize=tuple_size)
fig.subplots_adjust(left=0.2)
fig.subplots_adjust(bottom=0.2)
fig.subplots_adjust(top=0.99)
fig.subplots_adjust(right=0.96)

# -----------


def _find_optimal_reg_param_gen_error(
    alpha, var_func, var_hat_func, initial_cond, var_hat_kwargs, inital_value
):
    def minimize_fun(reg_param):
        m, q, _ = fp.state_equations(
            var_func,
            var_hat_func,
            reg_param=reg_param,
            alpha=alpha,
            init=initial_cond,
            var_hat_kwargs=var_hat_kwargs,
        )
        return 1 + q - 2 * m

    bnds = [(SMALLEST_REG_PARAM, None)]
    obj = minimize(
        minimize_fun,
        x0=inital_value,
        method="Nelder-Mead",
        bounds=bnds,
        options={"xatol": XATOL, "fatol": FATOL},
    )  # , , "maxiter":MAX_ITER
    if obj.success:
        fun_val = obj.fun
        reg_param_opt = obj.x
        return fun_val, reg_param_opt
    else:
        raise RuntimeError("Minima could not be found.")


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
        return fun_val, reg_param_opt, a_opt
    else:
        raise RuntimeError("Minima could not be found.")


# -------------------

N = 500
# epsilons = np.linspace(0.0, 0.5, N)
# epsilons = np.logspace(-4, np.log10(0.5), N)
epsilons = np.logspace(-4, np.log10(5e-1), N)

# N = int(N/2)

l2_err = np.empty(len(epsilons))
l2_lambda = np.empty(len(epsilons))

l1_err = np.empty(len(epsilons))
l1_lambda = np.empty(len(epsilons))

huber_err = np.empty(len(epsilons))
hub_lambda = np.empty(len(epsilons))
a_hub = np.empty(len(epsilons))

bo_err = np.empty(len(epsilons))

difference_hub = np.empty(len(epsilons))
difference_l2 = np.empty(len(epsilons))

print("First forward")
previous_aopt_hub = 1.171965327521171929e+01
previous_rpopt_hub = 9.999972243765241353e-02
previous_rpopt_l2 = 9.999999940395312703e-02
previous_rpopt_l1 = 3.894339831954845010e-01

for idx, eps in enumerate(epsilons):
    if idx >= 453:
        break
    print(eps)
    while True:
        m = 0.89 * np.random.random() + 0.1
        q = 0.89 * np.random.random() + 0.1
        sigma = 0.89 * np.random.random() + 0.1
        if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
            initial_condition = [m, q, sigma]
            break

    params = {
        "delta_small": delta_small,
        "delta_large": delta_large,
        "percentage": float(eps),
        "beta": beta,
    }

    l2_err[idx], l2_lambda[idx] = _find_optimal_reg_param_gen_error(
        alpha_cut,
        var_func_L2,
        var_hat_func_L2_decorrelated_noise,
        initial_condition,
        params,
        previous_rpopt_l2,
    )
    previous_rpopt_l2 = l2_lambda[idx]
    print("l2 reg {:.2f}".format(l2_lambda[idx]))

    l1_err[idx], l1_lambda[idx] = _find_optimal_reg_param_gen_error(
        alpha_cut,
        var_func_L2,
        var_hat_func_L1_decorrelated_noise,
        initial_condition,
        params,
        previous_rpopt_l1,
    )
    previous_rpopt_l1 = l1_lambda[idx]
    print("l1 reg {:.2f}".format(l1_lambda[idx]))

    if eps < 0.001:
        aaa = 10
    else:
        aaa = 1
    (
        huber_err[idx],
        hub_lambda[idx],
        a_hub[idx],
    ) = _find_optimal_reg_param_and_huber_parameter_gen_error(
        alpha_cut,
        var_hat_func_Huber_decorrelated_noise,
        initial_condition,
        params,
        [previous_rpopt_hub, previous_aopt_hub],
    )
    previous_aopt_hub = a_hub[idx]
    previous_rpopt_hub = hub_lambda[idx]
    print("hub reg {:.2f} hub a {:.2f}".format(hub_lambda[idx], a_hub[idx]))

    if eps == 0.0:
        ppp = {
            "delta": delta_small,
        }
        m, q, sigma = fp._find_fixed_point(
            alpha_cut, var_func_BO, var_hat_func_BO_single_noise, 1.0, initial_condition, ppp
        )
    else:
        pup = {
            "delta_small": delta_small,
            "delta_large": delta_large,
            "percentage": float(eps),
            "beta": beta,
        }
        m, q, sigma = fp._find_fixed_point(
            alpha_cut,
            var_func_BO,
            var_hat_func_BO_num_decorrelated_noise,
            1.0,
            initial_condition,
            pup,
        )
    bo_err[idx] = 1 - 2 * m + q
    print("done bo {}".format(idx))

    difference_hub[idx] = huber_err[idx] - bo_err[idx]
    difference_l2[idx] = l2_err[idx] - bo_err[idx]

print("Then backward")

previous_aopt_hub = 1.282544998216252896e-01
previous_rpopt_hub = 1.000000000000000021e-08
previous_rpopt_l2 = 1.371277809059197716e+00
previous_rpopt_l1 = 1.000000000000000021e-08

for idx, eps in enumerate(epsilons[::-1]):
    if idx >= 48:
        break
    print(eps)
    while True:
        m = 0.89 * np.random.random() + 0.1
        q = 0.89 * np.random.random() + 0.1
        sigma = 0.89 * np.random.random() + 0.1
        if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
            initial_condition = [m, q, sigma]
            break

    params = {
        "delta_small": delta_small,
        "delta_large": delta_large,
        "percentage": float(eps),
        "beta": beta,
    }

    l2_err[N-1-idx], l2_lambda[N-1-idx] = _find_optimal_reg_param_gen_error(
        alpha_cut,
        var_func_L2,
        var_hat_func_L2_decorrelated_noise,
        initial_condition,
        params,
        previous_rpopt_l2,
    )
    previous_rpopt_l2 = l2_lambda[N-1-idx]
    print("l2 reg {:.2f}".format(l2_lambda[N-1-idx]))

    l1_err[N-1-idx], l1_lambda[N-1-idx] = _find_optimal_reg_param_gen_error(
        alpha_cut,
        var_func_L2,
        var_hat_func_L1_decorrelated_noise,
        initial_condition,
        params,
        previous_rpopt_l1,
    )
    previous_rpopt_l1 = l1_lambda[N-1-idx]
    print("l1 reg {:.2f}".format(l1_lambda[N-1-idx]))

    if eps < 0.001:
        aaa = 10
    else:
        aaa = 1
    (
        huber_err[N-1-idx],
        hub_lambda[N-1-idx],
        a_hub[N-1-idx],
    ) = _find_optimal_reg_param_and_huber_parameter_gen_error(
        alpha_cut,
        var_hat_func_Huber_decorrelated_noise,
        initial_condition,
        params,
        [previous_rpopt_hub, previous_aopt_hub],
    )
    previous_aopt_hub = a_hub[N-1-idx]
    previous_rpopt_hub = hub_lambda[N-1-idx]
    print("hub reg {:.2f} hub a {:.2f}".format(hub_lambda[N-1-idx], a_hub[N-1-idx]))

    if eps == 0.0:
        ppp = {
            "delta": delta_small,
        }
        m, q, sigma = fp._find_fixed_point(
            alpha_cut, var_func_BO, var_hat_func_BO_single_noise, 1.0, initial_condition, ppp
        )
    else:
        pup = {
            "delta_small": delta_small,
            "delta_large": delta_large,
            "percentage": float(eps),
            "beta": beta,
        }
        m, q, sigma = fp._find_fixed_point(
            alpha_cut,
            var_func_BO,
            var_hat_func_BO_num_decorrelated_noise,
            1.0,
            initial_condition,
            pup,
        )
    bo_err[N-1-idx] = 1 - 2 * m + q
    print("done bo {}".format(N-1-idx))

    difference_hub[N-1-idx] = huber_err[N-1-idx] - bo_err[N-1-idx]
    difference_l2[N-1-idx] = l2_err[N-1-idx] - bo_err[N-1-idx]

np.savetxt(
    "./data/sweep_eps_fixed_delta_{:.2f}_beta_{:.2f}_alpha_cut_{:.2f}.csv".format(
        delta_large, beta, alpha_cut
    ),
    np.vstack((epsilons, l2_err, l1_err, huber_err, bo_err)).T,
    delimiter=",",
    header="epsilons,l2,l1,Huber,BO",
)

print(a_hub)

# ax.plot(epsilons, l2_err - bo_err, label=r"$\ell_2$")
# ax.plot(epsilons, l1_err - bo_err, label=r"$\ell_1$")
# ax.plot(epsilons, huber_err - bo_err, label="Huber")
# # ax.plot(epsilons, bo_err, label="BO")

# # ax.plot(epsilons, l2_err, label=r"$\ell_2$")
# # ax.plot(epsilons, l1_err, label=r"$\ell_1$")
# # ax.plot(epsilons, huber_err, label="Huber")
# # ax.plot(epsilons, bo_err, label="BO")
# ax.set_ylabel(r"$E_{\text{gen}} - E_{\text{gen}}^{BO}$")
# # ax.set_ylabel(r"$E_{\text{gen}}$")
# ax.set_xlabel(r"$\epsilon$")
ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlim([0.0001, 0.5])
# # ax.set_ylim([0.009, 1.5])
# ax.legend(ncol=2)

# ax.set_xticks([0.0001, 0.001, 0.01, 0.1, 0.5])
# ax.set_xticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$0.5$"])

# def fun(x, a, b):
#     return a*x+b

# popt, pcov = curve_fit(fun, np.log10(epsilons), np.log10(difference_hub), [1.0, -5])

# print(popt)
# print(pcov)

# string_to_annotate = "{:.2f}".format(popt[0]) + "\pm" + "{:.2f}".format(np.sqrt(pcov[0,0]))
# exponent = 0.898
# ax.annotate(r"$0.898 \pm 0.001$", (1e-3,1e-5))

# x = np.logspace(5e-4,5e-5,20)
# y=x**exponent
# ax.plot(x,y,'--')


# ax.plot(np.sqrt(np.log10(1/epsilons)), a_hub, label="Huber")
ax.plot(epsilons, l2_lambda, label="L2")
ax.plot(epsilons, l1_lambda, label="L1")
ax.plot(epsilons, hub_lambda, label="Hube")
ax.plot(epsilons, a_hub, label="a Hube")

# ax.set_ylabel(r"$a_{\text{opt}}$")
# ax.set_ylabel(r"$E_{\text{gen}}$")
ax.set_xlabel(r"$\epsilon$")
ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlim([0.0001, 0.5])
# ax.set_ylim([0.009, 1.5])
ax.legend(ncol=2)


np.savetxt(
    "./data/sweep_epsilon_bounded_figure_good_log.csv",
    np.vstack(
        (
            epsilons,
            l2_err,
            l2_lambda,
            l1_err,
            l1_lambda,
            huber_err,
            hub_lambda,
            a_hub,
            bo_err,
        )
    ).T,
    delimiter=",",
    header="# epsilon, errors_L2, lambdas_L2, errors_L1, lambdas_L1, errors_Huber,lambdas_Huber, huber_params, errors_BO",
)

if save:
    pu.save_plot(
        fig,
        "a_hub_sweep_eps_fixed_delta_{:.2f}_beta_{:.2f}_alpha_cut_{:.2f}".format(
            delta_large, beta, alpha_cut
        ),
    )

plt.show()

# plt.plot(np.log10(epsilons), np.log10(difference_hub))
# plt.show()
