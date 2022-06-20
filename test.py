import numpy as np
from numba import njit
from scipy.integrate import dblquad
import src.integration_utils as iu
import src.numerical_functions as num
from src.root_finding import brent_root_finder

XTOL = 1e-15
RTOL = 1e-11
EPSABS = 1e-9
EPSREL = 1e-9


@njit(error_model="numpy", fastmath=True)
def fun(x, a, b):
    return a * x + b


if __name__ == "__main__":
    q, m, sigma = 0.99, 0.01, 0.4
    delta_small, delta_large, eps = 0.1, 10.0, 0.3

    borders = iu.find_integration_borders_square(
        lambda y, xi: num.q_integral_BO_double_noise(
            y, xi, q, m, sigma, delta_small, delta_large, eps
        ),
        np.sqrt((1 + delta_small)),
        1.0,
    )
    print(borders)

    print(
        dblquad(
            num.q_integral_BO_double_noise,
            borders[0][0],
            borders[0][1],
            borders[1][0],
            borders[1][1],
            args=(q, m, sigma, delta_small, delta_large, eps),
            epsabs=EPSABS,
            epsrel=EPSREL,
        )[0]
    )

