import numpy as np
from numba import njit
from src.root_finding import brent_root_finder

XTOL = 1e-15
RTOL = 1e-11


@njit(error_model="numpy", fastmath=True)
def fun(x, a, b):
    return a * x + b


if __name__ == "__main__":
    print(brent_root_finder(fun, -20, 20, XTOL, RTOL, 10000, (-1, np.pi)))

