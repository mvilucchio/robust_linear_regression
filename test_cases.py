import numerical_functions as num
import fixed_point_equations as fpe
import numpy as np
import unittest
import math

class TestNumericalFunctions(unittest.TestCase):
    def test_equality_funcs(self):
        delta = 1.0
        alpha = 1.0

        self.assertTrue()

if __name__ == '__main__':
    # unittest.main()
    delta = 1.0
    alpha = 1.0

    m, q, sigma = 0.0, 0.0, 0.0
    while True:
        m = np.random.random()
        q = np.random.random()
        sigma = np.random.random()
        if np.square(m) < q + delta * q:
            break
    
    anal_m_hat, anal_q_hat, anal_sigma_hat = fpe.var_hat_func_L2(m, q, sigma, alpha, delta)
    num_m_hat, num_q_hat, num_sigma_hat = fpe.var_hat_func_L2_num(m, q, sigma, alpha, delta)

    print(anal_m_hat, anal_q_hat, anal_sigma_hat)
    print(num_m_hat, num_q_hat, num_sigma_hat)

    print(anal_m_hat - num_m_hat, anal_q_hat - num_q_hat, anal_sigma_hat - num_sigma_hat)

    print(math.isclose(anal_m_hat, num_m_hat), math.isclose(anal_q_hat, num_q_hat), math.isclose(anal_sigma_hat, num_sigma_hat))
