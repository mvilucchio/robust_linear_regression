import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
from tqdm.auto import tqdm
import fixed_point_equations as fpe
from mpl_toolkits.mplot3d import Axes3D
import numerical_functions as num
import numerical_function_double as numdub
from scipy.integrate import dblquad
from matplotlib import cm


if __name__ == "__main__":
    delta_small = 0.1
    delta_large = 2.0
    eps = 0.1

    for i in range(1):
        print("i = {} ".format(i))
        while True:
            m, q, sigma = 0.89 * np.random.random(size=3) + 0.1
            if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
                break
        a = 1.0

        q = 0.2
        m = q
        sigma = 1 - q

        # print("m = {}; q = {}; sigma= {};".format(m, q, sigma))

        borders = numdub.find_integration_borders(
            lambda y, xi: numdub.q_integral_BO_eps(
                y, xi, q, m, sigma, delta_small, delta_large, eps
            ),
            np.sqrt((1 + delta_small)),
            1.0,
        )

        range_max = borders[0][1]
        print("range max {}".format(range_max))

        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # # Make data.
        # X = np.linspace(-range_max, range_max, 300)
        # Y = np.linspace(-range_max, range_max, 300)
        # X, Y = np.meshgrid(X, Y)
        # Z = np.empty_like(X)
        # for i in range(len(X[0])):
        #     for j in range(len(X[0])):
        #         Z[i, j] = numdub.q_integral_BO_eps(
        #             Y[i, j], X[i, j], q, m, sigma, delta_small, delta_large, eps
        #         )

        # # Plot the surface.
        # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

        # # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        # ax.set_xlabel(r"$\xi$")
        # ax.set_ylabel(r"$y$")

        # plt.show()

        # int_val_1 = numdub.integral_fpe(
        #     numdub.m_integral_Huber_eps,
        #     numdub.border_plus_Huber,
        #     numdub.border_minus_Huber,
        #     numdub.test_fun_upper_Huber,
        #     m,
        #     q,
        #     sigma,
        #     delta_small,
        #     delta_large,
        #     eps,
        # )

        # int_val_2 = numdub.integral_fpe(
        #     numdub.q_integral_Huber_eps,
        #     numdub.border_plus_Huber,
        #     numdub.border_minus_Huber,
        #     numdub.test_fun_upper_Huber,
        #     m,
        #     q,
        #     sigma,
        #     delta_small,
        #     delta_large,
        #     eps,
        # )

        # int_val_3 = numdub.integral_fpe(
        #     numdub.sigma_integral_Huber_eps,
        #     numdub.border_plus_Huber,
        #     numdub.border_minus_Huber,
        #     numdub.test_fun_upper_Huber,
        #     m,
        #     q,
        #     sigma,
        #     delta_small,
        #     delta_large,
        #     eps,
        # )

        int_val_2 = numdub.q_hat_equation_BO_eps(
            m, q, sigma, delta_small, delta_large, eps
        )

        # if np.any(np.isnan([int_val_1, int_val_2, int_val_3])):
        #     raise ValueError("is nan")

        print("q_int : {}".format(int_val_2))

        # xi = np.linspace(-range_max, range_max, 100)

        # plt.figure(figsize=(6, 6))
        # plt.plot(
        #     xi, numdub.border_plus_Huber(xi, m, q, sigma), label="plus",
        # )
        # plt.plot(
        #     xi, numdub.border_minus_Huber(xi, m, q, sigma), label="minus",
        # )

        # print(numdub.border_plus_Huber(-range_max, m, q, sigma))

        # plt.xlabel(r"$\xi$")
        # plt.ylabel(r"$y$")
        # plt.ylim([-range_max, range_max])
        # plt.xlim([-range_max, range_max])
        # plt.legend()
        # plt.grid()
        # plt.show()
