import matplotlib.pyplot as plt
import numpy as np
import src.fpeqs as fp
from scipy.special import erf
from src.fpeqs import different_alpha_observables_fpeqs
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


delta_small = 0.1
delta_large = 10.0
beta = 0.0
# a = 10
alpha_cut = 1000000
reg_param = 1.0
N = 10
a_hub = 1.0

epsilons = np.linspace(0.0, 0.5, N)
ms_l2 = np.empty_like(epsilons)
qs_l2 = np.empty_like(epsilons)

# ms_l1 = np.empty_like(epsilons)
# qs_l1 = np.empty_like(epsilons)

ms_hub = np.empty_like(epsilons)
qs_hub = np.empty_like(epsilons)

for idx, eps in enumerate(epsilons):

    while True:
        m = 0.89 * np.random.random() + 0.1
        q = 0.89 * np.random.random() + 0.1
        sigma = 0.89 * np.random.random() + 0.1
        if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
            initial_condition = [m, q, sigma]
            break

    pup = {
        "delta_small": delta_small,
        "delta_large": delta_large,
        "percentage": float(eps),
        "beta": beta,
        "a": a_hub,
    }

    # ms_l1[idx], qs_l1[idx], _ = fp._find_fixed_point(
    #     alpha_cut,
    #     var_func_L2,
    #     var_hat_func_L1_decorrelated_noise,
    #     reg_param,
    #     initial_condition,
    #     pup,
    # )

    ms_hub[idx], qs_hub[idx], _ = fp._find_fixed_point(
        alpha_cut,
        var_func_L2,
        var_hat_func_Huber_decorrelated_noise,
        reg_param,
        initial_condition,
        pup,
    )

    pup = {
        "delta_small": delta_small,
        "delta_large": delta_large,
        "percentage": float(eps),
        "beta": beta,
        # "a": a_hub,
    }

    ms_l2[idx], qs_l2[idx], _ = fp._find_fixed_point(
        alpha_cut,
        var_func_L2,
        var_hat_func_L2_decorrelated_noise,
        reg_param,
        initial_condition,
        pup,
    )



egen_l2 = 1 + qs_l2 - 2 * ms_l2
# egen_l1 = 1 + qs_l1 - 2 * ms_l1
egen_hub = 1 + qs_hub - 2 * ms_hub

# egen_frac = egen_l1 / egen_l2
egen_frac = egen_hub / egen_l2

for idx in range(len(epsilons)):
    print(epsilons[idx], egen_frac[idx])

plt.plot(epsilons, egen_frac, marker='.')
# plt.plot(epsilons, qs, label="q", marker='.')
# plt.xscale("log")
plt.yscale("log")
# plt.ylim([0.1, 1.5])
plt.xlim([0.0, 0.5])
plt.grid()
plt.xlabel(r"$\epsilon$")
plt.ylabel(r"$E_{gen,Hub} / E_{gen,\ell_2}$")
# plt.legend()

# for idx_e, vertical_pos in zip([2, 20, 30, 50], [0.1, 0.2, 0.3, 0.4]):
#     plt.annotate(
#         "({:.4f},{:.4f})".format(epsilons[idx_e], egen_frac[idx_e]),
#         xy=(idx_e / N, vertical_pos),
#         xytext=(idx_e / N, vertical_pos),
#         textcoords="axes fraction",
#     )
#     plt.axvline(epsilons[idx_e], alpha=0.5)

plt.annotate(
    r"$(\epsilon,E_{gen,Hub} / E_{gen,\ell_2})$",
    (0.2, 0.9),
    xytext=(0.2, 0.9),
    textcoords="figure fraction",
)

plt.show()
