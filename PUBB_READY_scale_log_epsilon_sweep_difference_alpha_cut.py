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

tuple_size = pu.set_size(width, fraction=0.50)

fig, ax = plt.subplots(1, 1, figsize=tuple_size)
fig.subplots_adjust(left=0.16)
fig.subplots_adjust(bottom=0.2283)
fig.subplots_adjust(top=0.91)
fig.subplots_adjust(right=0.95)

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

N=40
epsilons_inset = np.logspace(-6, -3, N)
huber_err_inset = np.empty(len(epsilons_inset))
a_hub_inset = np.empty(len(epsilons_inset))
bo_err_inset = np.empty(len(epsilons_inset))

difference_hub_inset = np.empty(len(epsilons_inset))

for idx, eps in enumerate(tqdm(epsilons_inset)):

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

    if eps < 0.001:
        aaa = 10
    else:
        aaa = 1

    huber_err_inset[idx], _, a_hub_inset[idx] = _find_optimal_reg_param_and_huber_parameter_gen_error(
        alpha_cut,
        var_hat_func_Huber_decorrelated_noise,
        initial_condition,
        params,
        [0.01, aaa],
    )

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
    bo_err_inset[idx] = 1 - 2 * m + q

    difference_hub_inset[idx] = huber_err_inset[idx] - bo_err_inset[idx]

# width, hieght = pu.set_size(0.5, fraction=0.49)


ax.plot(epsilons_inset * np.log(1 / epsilons_inset), difference_hub_inset, color='tab:orange', label="Huber", marker='.', linewidth=0.0)
ax.set_xlabel(r"$ -\epsilon \log(\epsilon)$")
ax.set_ylabel(r"$E_{\text{gen}}^{\text{Huber}} - E_{\text{gen}}^{\text{BO}}$")

def fun(x, a, b):
    return x * a + b

popt, pcov = curve_fit(fun, epsilons_inset * np.log(1 / epsilons_inset), difference_hub_inset, p0=[0.01, 0.01])

print("fit done")
print(popt)
print(pcov)

xs = np.linspace(0.0, 0.007, 30)
ax.plot(xs, fun(xs, *popt), label="Linear Fit", color='black', linestyle='solid')
ax.set_xlim([0.0, 0.007])

if save:
    pu.save_plot(
        fig,
        "sweep_eps_scaling_fixed_delta_{:.2f}_beta_{:.2f}_alpha_cut_{:.2f}_delta_small_{:.2f}".format(  # "a_hub_sweep_eps_fixed_delta_{:.2f}_beta_{:.2f}_alpha_cut_{:.2f}".format( # "sweep_eps_fixed_delta_{:.2f}_beta_{:.2f}_alpha_cut_{:.2f}".format(
            delta_large, beta, alpha_cut, delta_small
        ),
    )

plt.show()