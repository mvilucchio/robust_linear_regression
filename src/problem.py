import numpy as np
import numba as nb
from utils import find_integration_borders_square


class Problem:
    def var_functions(
        self, m_hat, q_hat, sigma_hat, alpha, delta_small, delta_large, eps, reg_param
    ):
        pass

    def var_hat_functions(self, m, q, sigma, alpha, delta_small, delta_large, eps):
        pass

    def get_loss_name(self):
        pass


class GroundUpProblem(Problem):
    def __init__(self, loss, prior):
        self.loss = loss
        self.prior = prior
        return

    def m_integral_L2(y, xi, q, m, sigma, delta_small, delta_large, eps):
        pass

    def q_integral_L2(y, xi, q, m, sigma, delta_small, delta_large, eps):
        pass

    def sigma_integral_L2(y, xi, q, m, sigma, delta_small, delta_large, eps):
        pass

    def var_functions(
        self, m_hat, q_hat, sigma_hat, alpha, delta_small, delta_large, eps, reg_param
    ):
        pass

    def var_hat_functions(self, m, q, sigma, alpha, delta_small, delta_large, eps):
        pass

    def get_loss_name(self):
        pass


class FullyDefinedProblem(Problem):
    def m_integral_L2(y, xi, q, m, sigma, delta_small, delta_large, eps):
        pass

    def q_integral_L2(y, xi, q, m, sigma, delta_small, delta_large, eps):
        pass

    def sigma_integral_L2(y, xi, q, m, sigma, delta_small, delta_large, eps):
        pass

    def var_functions(
        self, m_hat, q_hat, sigma_hat, alpha, delta_small, delta_large, lambd
    ):
        pass

    def var_hat_functions(self, m, q, sigma, alpha, delta_small, delta_large):
        pass

    def get_loss_name(self):
        pass


@nb.jitclass()
class RidgeProblemAnalytical(FullyDefinedProblem):
    @staticmethod
    def var_functions(
        self, m_hat, q_hat, sigma_hat, alpha, delta_small, delta_large, reg_param
    ):
        m = m_hat / (sigma_hat + reg_param)
        q = (np.square(m_hat) + q_hat) / np.square(sigma_hat + reg_param)
        sigma = 1.0 / (sigma_hat + reg_param)
        return m, q, sigma

    @staticmethod
    def var_hat_functions(self, m, q, sigma, alpha, delta_small, delta_large):
        m_hat = alpha / (1 + sigma)
        q_hat = alpha * (1 + q + delta_small - 2 * np.abs(m)) / ((1 + sigma) ** 2)
        sigma_hat = alpha / (1 + sigma)
        return m_hat, q_hat, sigma_hat

    @staticmethod
    def get_loss_name(self):
        return "L2"


class RidgeProblemNumerical(FullyDefinedProblem):
    def var_functions(
        self, m_hat, q_hat, sigma_hat, alpha, delta_small, delta_large, eps, reg_param
    ):
        m = m_hat / (sigma_hat + reg_param)
        q = (np.square(m_hat) + q_hat) / np.square(sigma_hat + reg_param)
        sigma = 1.0 / (sigma_hat + reg_param)
        return m, q, sigma

    def var_hat_functions(self, m, q, sigma, alpha, delta_small, delta_large, eps):
        pass

    def get_loss_name(self):
        return "L2"
