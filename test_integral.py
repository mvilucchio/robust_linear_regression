from cProfile import label
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
    delta_small = 1.0
    delta_large = 10.0
    eps = 0.1

    m, q, sigma, a, alpha = 0.9, 0.9, 1.1, 1, 1

    borders = numdub.find_integration_borders(
        lambda y, xi: numdub.m_integral_L2_eps(y, xi, q, m, sigma, delta_small, delta_large, eps), 
        np.sqrt((1 + delta_small)), 
        1.0
    )

    print(borders)
    
    range_max = borders[0][1]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.linspace(-range_max, range_max, 300)
    Y = np.linspace(-range_max, range_max, 300)
    X, Y = np.meshgrid(X, Y)
    Z = np.empty_like(X)
    for i in range(len(X[0])):
        for j in range(len(X[0])):
            Z[i,j] = numdub.m_integral_L2_eps(Y[i,j], X[i,j], q, m, sigma, delta_small, delta_large, eps)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$y$")

    plt.show()


    # if sigma + 1 >= 0:
    #     print("sigma + 1 >= 0 integral : {:.9f}".format(
    #         num.integral_fpe(num.m_integral_Huber,
    #                          num.border_plus, num.border_minus, num.test_fun_upper, m, q, sigma, delta)
    #     ))
    # else:
    #     print("sigma + 1 < 0 integral : {:.9f}".format(
    #         num.integral_fpe(num.m_integral_Huber,
    #                          num.border_minus, num.border_plus, num.test_fun_down, m, q, sigma, delta)
    #     ))

    # xi = np.linspace(-8, 8, 100)

    # plt.plot(xi, num.border_plus(xi, m, q, sigma, delta), label="plus")
    # plt.plot(xi, num.border_minus(xi, m, q, sigma, delta), label="minus")

    # plt.ylim([-7.0710678118654755, 7.0710678118654755])
    # plt.xlim([-7.0710678118654755, 7.0710678118654755])
    # plt.legend()
    # plt.show()
