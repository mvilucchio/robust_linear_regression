from cv2 import integral
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from tqdm.auto import tqdm
from scipy.integrate import dblquad, quad
import fixed_point_equations as fpe

MULT_INTEGRAL = 10

def ZoutBayes(y, omega, V, delta):
    return np.exp(-(y - omega)**2 / ( 2 * (V + delta) ) ) / np.sqrt( 2*np.pi * (V + delta) )

def foutBayes(y, omega, V, delta):
    return (y - omega) / (V + delta)

def foutL2(y, omega, V):
    return (y - omega) / (1 + V)

def DfoutL2(y, omega, V):
    return - 1.0 / (1 + V)

# def foutL1():
#     return

# def DfoutL1():
#     return

def find_integration_borders(fun, scale1, scale2, mult = MULT_INTEGRAL, tol=1e-8, n_points=300):
    borders = [[- mult * scale1, mult * scale1], [- mult * scale2, mult * scale2]]

    for idx, ax in enumerate(borders):
        for jdx, border in enumerate(ax):

            while True:
                if idx == 0:
                    max_val = np.max(
                        fun(
                            borders[idx][jdx],
                            np.linspace(borders[1 if idx == 0 else 0][0], borders[1 if idx == 0 else 0][1], n_points)
                        )
                    )
                else:
                    max_val = np.max(
                        fun(
                            np.linspace(borders[1 if idx == 0 else 0][0], borders[1 if idx == 0 else 0][1], n_points),
                            borders[idx][jdx]
                        )
                    )
                if max_val > tol:
                    borders[idx][jdx] = borders[idx][jdx] + (-1.0 if jdx == 0 else 1.0) * (scale1 if idx == 0 else scale2)
                else:
                    break

    return borders

def integral1(y, xi, q, m, sigma, delta):
    eta = m**2 / q
    return np.exp(- xi**2 / 2) / np.sqrt(2 * np.pi) * ZoutBayes(y, np.sqrt(eta) * xi, (1 - eta), delta) * foutBayes(y, np.sqrt(eta) * xi, (1 - eta), delta) * foutL2(y, np.sqrt(q) * xi, sigma)

def integral2(y, xi, q, m, sigma, delta):
    eta = m**2 / 2
    return np.exp(- xi**2 / 2) / np.sqrt(2 * np.pi) * ZoutBayes(y, np.sqrt(eta) * xi, (1 - eta), delta) * foutL2(y, np.sqrt(q) * xi, sigma) * foutL2(y, np.sqrt(q) * xi, sigma)

def integral3(y, xi, q, m, sigma, delta):
    eta = m**2 / 2
    return np.exp(- xi**2 / 2) / np.sqrt(2 * np.pi) * ZoutBayes(y, np.sqrt(eta) * xi, (1 - eta), delta) * DfoutL2(y, np.sqrt(q) * xi, sigma)

def m_hat_equation_L2(m, q, sigma, delta):
    borders = find_integration_borders(
        lambda y, xi : integral1(y, xi, q, m, sigma, delta), 
        np.sqrt((1 + delta)),
        1.0
    )
    return dblquad(
        lambda y, xi: integral1(y, xi, q, m, sigma, delta), 
        borders[0][0], 
        borders[0][1], 
        borders[1][0],
        borders[1][1]
    )[0]

def q_hat_equation_L2(m, q, sigma, delta):
    borders = find_integration_borders(
        lambda y, xi : integral2(y, xi, q, m, sigma, delta), 
        np.sqrt((1 + delta)),
        1.0
    )
    return dblquad(
        lambda y, xi: integral2(y, xi, q, m, sigma, delta), 
        borders[0][0], 
        borders[0][1], 
        borders[1][0],
        borders[1][1] # lambda xi: np.sqrt(eta) * xi - size_y, lambda xi: np.sqrt(eta) * xi + size_y
    )[0]

def sigma_hat_equation_L2(m, q, sigma, delta):
    borders = find_integration_borders(
        lambda y, xi : integral3(y, xi, q, m, sigma, delta), 
        np.sqrt((1 + delta)),
        1.0
    )
    return dblquad(
        lambda y, xi: integral3(y, xi, q, m, sigma, delta), 
        borders[0][0], 
        borders[0][1], 
        borders[1][0],
        borders[1][1]
    )[0]

def state_equations_convergence(var_func, var_hat_func, delta = 0.1, lamb = 0.1, alpha = 0.5, init=(.5, .4, 1), verbose=False):
    m, q, sigma = init[0], init[1], init[2]
    err = 1.0
    blend = 0.6
    iter = 0
    while err > 1e-6:
        m_hat, q_hat, sigma_hat = var_hat_func(m, q, sigma, alpha, delta)

        temp_m = m
        temp_q = q
        temp_sigma = sigma

        m, q, sigma = var_func(m_hat, q_hat, sigma_hat, alpha, delta, lamb)

        err = np.max(np.abs([temp_m - m, temp_q - q, temp_sigma - sigma]))

        m = blend*m + (1-blend)*temp_m
        q = blend*q + (1-blend)*temp_q
        sigma = blend*sigma + (1-blend)*temp_sigma
        if verbose:
            print(f'i = {iter} m = {m}, q = {q}, sigma = {sigma}, eta = {m**2/q}')
        iter += 1
    return m, q, sigma

if __name__ == "__main__":
    # test the convergence
    alpha = 0.01
    deltas = [1.0]
    lambdas = [1.0]

    for idx, l in enumerate(tqdm(lambdas, desc="lambda", leave=False)):
        for jdx, delta in enumerate(tqdm(deltas, desc="delta", leave=False)):
            i = idx * len(deltas) +  jdx

            while True:
                m = np.random.random()
                q = np.random.random()
                sigma = np.random.random()
                if np.square(m) < q + delta * q:
                    break

            initial = [m, q, sigma]

            _, _, _ = state_equations_convergence(
                fpe.var_func_L2, 
                fpe.var_hat_func_L2_num, 
                delta = delta, 
                lamb = l, 
                alpha = alpha, 
                init = initial, 
                verbose = True
            )