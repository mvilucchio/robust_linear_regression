import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
import numerical_functions as numfun
from tqdm.auto import tqdm


class FixedPointFinder:
    def __init__(self, loss, condition=None, verbose=False, save_state=False, file_name="./fixed_point.npy"):
        self.loss = loss
        self.condition = condition
        self.verbose = verbose
        self.save_state = save_state
        self.file_name = file_name
        self.all_alphas = {}
        self.all_fixed_points = {}

    def find_fixed_point_curve(self, alpha_min, alpha_max, n_alpha_pts, save=None, file_name=None):
        self.alphas = np.logspace(np.log(alpha_min) / np.log(10), np.log(alpha_max) / np.log(10), n_alpha_pts)
        self.fixed_points = np.zeros((n_alpha_pts, 3))

        if self.condition is None:
            initial_pts = np.random.random(size=(3,))
        else:
            while True:
                initial_pts = np.random.random(size=(3,))
                if self.condition(*initial_pts):
                    break

        for idx, alpha in enumerate(tqdm(self.alphas, desc="alpha", disable=not self.verbose, leave=False)):
            self.fixed_points[idx][0], self.fixed_points[idx][1], self.fixed_points[idx][2] = self.state_equations(
                var_func, var_hat_func, alpha, initial_pts
            )

        if save is None and self.save_state:
            if file_name is None:
                file_name = self.file_name
            np.savez(file_name, alpha=self.alphas, observables=self.fixed_points)

        # self.all_alphas[reg_param] =
        # self.all_fixed_points[reg_param] =

    def state_equation_iteration(
        self,
        var_func,
        var_hat_func,
        alpha,
        init,
        delta,
        reg_param,
        blend=0.1,
        abs_tol=1e-03,
        rel_tol=1e-03,
    ):
        m, q, sigma = init[0], init[1], init[2]
        err_rel = 1.0
        err_abs = 1.0
        while err_rel > rel_tol and err_abs > abs_tol:
            m_hat, q_hat, sigma_hat = var_hat_func(m, q, sigma, alpha, delta)

            temp_m, temp_q, temp_sigma = m, q, sigma

            m, q, sigma = var_func(m_hat, q_hat, sigma_hat, alpha, delta, reg_param)

            err_abs = np.max(np.abs([temp_m - m, temp_q - q, temp_sigma - sigma]))
            err_rel = np.max(
                np.abs(
                    [(temp_m - m) / m, (temp_q - q) / q, (temp_sigma - sigma) / sigma]
                )
            )

            m = blend * m + (1 - blend) * temp_m
            q = blend * q + (1 - blend) * temp_q
            sigma = blend * sigma + (1 - blend) * temp_sigma

        return m, q, sigma
