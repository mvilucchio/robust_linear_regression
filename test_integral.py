from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
from tqdm.auto import tqdm
import fixed_point_equations as fpe
from mpl_toolkits.mplot3d import Axes3D
import numerical_functions as num
from scipy.integrate import dblquad
from matplotlib import cm


if __name__ == "__main__":
    delta = 1.0

    m, q, sigma, a, alpha = 0.9, 3, -1.1, 1, 1

    borders = [[-10, 10], [-10, 10]]

    if sigma + 1 >= 0:
        print("sigma + 1 >= 0 integral : {:.9f}".format(
            num.integral_fpe(num.m_integral_Huber,
                             num.border_plus, num.border_minus, num.test_fun_upper, m, q, sigma, delta)
        ))
    else:
        print("sigma + 1 < 0 integral : {:.9f}".format(
            num.integral_fpe(num.m_integral_Huber,
                             num.border_minus, num.border_plus, num.test_fun_down, m, q, sigma, delta)
        ))

    xi = np.linspace(-8, 8, 100)

    plt.plot(xi, num.border_plus(xi, m, q, sigma, delta), label="plus")
    plt.plot(xi, num.border_minus(xi, m, q, sigma, delta), label="minus")

    plt.ylim([-7.0710678118654755, 7.0710678118654755])
    plt.xlim([-7.0710678118654755, 7.0710678118654755])
    plt.legend()
    plt.show()
