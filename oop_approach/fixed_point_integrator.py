import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
import numerical_functions as numfun
from tqdm.auto import tqdm

class FixedPointFinder():
    def __init__(self, loss, delta, condition=None, verbose=False):
        self.loss = loss
        self.delta = delta
        self.condition = condition
        self.verbose = verbose
    
    def find_fixed_point(self, alpha_min, alpha_max, n_alpha_pts):
        self.alphas = np.logspace(np.log(alpha_min)/np.log(10), np.log(alpha_max)/np.log(10), n_alpha_pts)
        self.fixed_points = np.zeros((n_alpha_pts, 3))

        if self.condition is None:
            initial_pts = np.random.random(size=(3,))
        else:
            while True:
                initial_pts = np.random.random(size=(3,))
                if self.condition(initial_pts):
                    break

        for idx, alpha in enumerate(tqdm(self.alphas, desc="alpha", disable=not self.verbose, leave=False)):
            self.fixed_points[idx][0], self.fixed_points[idx][1], self.fixed_points[idx][2] = self.state_equations(
                var_func, 
                var_hat_func, 
                alpha, 
                initial_pts
            )

    def state_equation(self, var_func, var_hat_func, alpha, init, blend = 0.7, tol = 1e-07):
        m, q, sigma = init[0], init[1], init[2]
        err = 1.0

        while err > tol:
            m_hat, q_hat, sigma_hat = var_hat_func(m, q, sigma, alpha, self.delta)

            temp_m, temp_q, temp_sigma = m, q, sigma

            m, q, sigma = var_func(m_hat, q_hat, sigma_hat, alpha, self.delta, self.loss.reg_param)

            err = np.min(np.abs([temp_m - m, temp_q - q, temp_sigma - sigma]))

            m = blend * m + (1 - blend)*temp_m
            q = blend * q + (1 - blend)*temp_q
            sigma = blend * sigma + (1 - blend) * temp_sigma

        return m, q, sigma