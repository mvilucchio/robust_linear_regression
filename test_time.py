import numpy as np
import numerical_functions as num
import time
import numba as nb

n_points = 300
borders = [[-3, 3], [-3, 3]]
idx = 0

other_args = (0.2, 0.0, 0.1, 1.0)

@nb.njit(error_model="numpy", fastmath=True)
def ZoutBayes(y, omega, V, delta):
    return np.exp(-(y - omega)**2 / ( 2 * (V + delta) ) ) / np.sqrt( 2*np.pi * (V + delta) )

@nb.njit(error_model="numpy", fastmath=True)
def foutBayes(y, omega, V, delta):
    return (y - omega) / (V + delta)

@nb.vectorize
def foutL1(y, omega, V):
    if omega <= y - V:
        return 1
    elif omega <= y + V:
        return (y - omega) / V
    else:
        return -1

@nb.njit(error_model="numpy", fastmath=True)
def m_integral_L1(y, xi, q, m, sigma, delta):
    eta = m**2 / q
    return np.exp(- xi**2 / 2) / np.sqrt(2 * np.pi) * \
        ZoutBayes(y, np.sqrt(eta) * xi, (1 - eta), delta) * \
        foutBayes(y, np.sqrt(eta) * xi, (1 - eta), delta) * \
        foutL1(y, np.sqrt(q) * xi, sigma)

start_1 = time.time()
max_val = np.max(
        m_integral_L1(
            -3,
            np.linspace(borders[1 if idx == 0 else 0][0], borders[1 if idx == 0 else 0][1], n_points),
            *other_args
        )
)
print(max_val)
end_1 = time.time()

start_2 = time.time()
max_val = np.max(
    [
        num.m_integral_L1(
            -3,
            pt,
            *other_args
        )
        for pt in np.linspace(borders[1 if idx == 0 else 0][0], borders[1 if idx == 0 else 0][1], n_points)
    ]
)
print(max_val)
end_2 = time.time()


print(end_1 - start_1)
print(end_2 - start_2)