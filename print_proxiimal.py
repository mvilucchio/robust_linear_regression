import matplotlib.pyplot as plt
import numpy as np
from src.fpeqs import (
    no_parallel_different_alpha_observables_fpeqs,
    different_alpha_observables_fpeqs,
    var_func_L2,
    var_hat_func_L2_decorrelated_noise,
    var_hat_func_L2_double_noise,
    var_hat_func_numerical_loss_single_noise,
)
import src.loss_functions as losses
from src.loss_functions import (
    proximal_loss_double_quad,
    _proximal_argument_derivative_loss_double_quad,
    _proximal_argument_loss_double_quad,
)

if __name__ == "__main__":

    y, omega, V, width = 0, 0, 1, 0.5
    y, omega, V = -10.48809, 0.37840, 7.18624
    loss_args = {"width": width}
    omegas = np.linspace(-1, 1, 501)

    zs = np.linspace(-15, 0, 100)
    plt.plot(zs, _proximal_argument_loss_double_quad(zs, y, omega, V, width))

    plt.plot(
        zs,
        np.array(
            [
                _proximal_argument_derivative_loss_double_quad(z, y, omega, V, width)
                for z in zs
            ]
        ),
    )

    pv, _ = proximal_loss_double_quad(y, omega, V, width)

    # aa = proximal_loss_double_quad(y, omega, V, width)
    # plt.axvline(x=aa)
    #  plt.plot(zs, fun_evals, ".")
    plt.axvline(x=(V * y + omega) / (1 + V) - 2 * V / (V + 1), color="g")
    plt.axvline(x=(V * y + omega) / (1 + V) + 2 * V / (V + 1), color="g")
    plt.axvline(x=y - 15 * width / V)
    plt.axvline(x=y + 15 * width / V)
    plt.axvline(x=pv, color="r")
    plt.grid(which="major")
    plt.grid(which="minor")
    plt.show()

    # prox = np.empty((len(omegas),))
    # for idx, o in enumerate(omegas):
    #     print("omega {}".format(o))
    #     prox[idx] = proximal_loss_double_quad(y, o, V, width)[0]

    # # #  plt.plot(omegas, losses.loss_l2(0, omegas))
    # plt.plot(omegas, prox, ".-", markersize=3, linewidth=0.5)
    # plt.grid()
    # plt.show()

    # delta_small = 0.1
    # delta_large = 10.0
    # percentage = 0.1
    # beta = 0.5

    # var_hat_kwargs = {
    #     "delta": delta_small,
    #     "proximal_func": losses.proximal_loss_double_quad,
    #     "loss_args": {"width": 0.15},
    # }

    # while True:
    #     m = 0.89 * np.random.random() + 0.1
    #     q = 0.89 * np.random.random() + 0.1
    #     sigma = 0.89 * np.random.random() + 0.1
    #     if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
    #         initial_condition = [m, q, sigma]
    #         break

    # alphas, [m, q, sigma] = no_parallel_different_alpha_observables_fpeqs(
    #     var_func_L2,
    #     var_hat_func_numerical_loss_single_noise,
    #     funs=[lambda m, q, sigma: m, lambda m, q, sigma: q, lambda m, q, sigma: sigma],
    #     alpha_1=0.01,
    #     alpha_2=100,
    #     n_alpha_points=4,
    #     reg_param=0.1,
    #     initial_cond=initial_condition,
    #     var_hat_kwargs=var_hat_kwargs,
    # )

    # plt.plot(alphas, m, label="m")
    # plt.plot(alphas, q, label="q")
    # plt.plot(alphas, sigma, label="sigma")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.grid()
    # plt.legend()

    # plt.show()
