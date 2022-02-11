import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
from tqdm.auto import tqdm

def q_fun(qhat):
    return qhat / (1 + qhat)

def qhat_fun(q, delta, alpha):
    return alpha / (1 + delta - q)

def state_equations_bayes_optimal(delta = .001, alpha = .5, init=[.4]):
    q = init[0]
    err = 1
    blend = .7
    while err > 1e-6:
        q_hat = qhat_fun(q, delta, alpha)
        temp_q = q
        q = q_fun(q_hat)
        err = np.abs(temp_q - q)
        q = blend * q + (1 - blend) * temp_q
    return q

def projection_bayes_optimal_different_alpha_theory(alpha_1=0.1, alpha_2=100, n_alpha_points=15, delta=1):
    
    alphas = np.logspace(np.log(alpha_1)/np.log(10), np.log(alpha_2)/np.log(10), n_alpha_points)

    init = [.4]
    error_theory = np.zeros(n_alpha_points)

    for i, alpha in enumerate(alphas):
        q = state_equations_bayes_optimal(delta, alpha, init)
        error_theory[i] = 1 + q - 2 * q
    
    return error_theory, alphas