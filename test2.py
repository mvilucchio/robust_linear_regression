import numpy as np
from numba import njit
from tqdm.auto import tqdm
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import src.integration_utils as iu
import src.numerical_functions as num
from src.root_finding import brent_root_finder
from src.amp_funcs import (
    input_functions_gaussian_prior,
    output_functions_decorrelated_noise,
    output_functions_double_noise,
    output_functions_single_noise,
)
from src.numerics import (
    find_coefficients_AMP,
    data_generation,
    measure_gen_single,
)

# XTOL = 1e-15
# RTOL = 1e-11
# EPSABS = 1e-9
# EPSREL = 1e-9

n = 100
d = 1000
delta = 0.5
delta_small = 0.5
delta_large = 3.0
eps = 0.1
beta = 0.0
n_alpha_points = 15

if __name__ == "__main__":

    errors_mean = np.empty((n_alpha_points,))
    errors_std = np.empty((n_alpha_points,))

    results = np.empty((n_alpha_points,))
    i = 0

    alphas = np.logspace(-0.5, 1, n_alpha_points)

    for idx, alpha in enumerate(tqdm(alphas)):
        xs, ys, _, _, ground_truth_theta = data_generation(
            measure_gen_single,
            n_features=d,
            n_samples=max(int(np.around(d * alpha)), 1),
            n_generalization=1,
            measure_fun_args=(1.0,),
        )

        a, v = find_coefficients_AMP(
            input_functions_gaussian_prior, output_functions_single_noise, ys, xs, delta,
        )

        results[idx] = np.divide(np.sum(np.square(ground_truth_theta - a)), d)

        del xs
        del ys
        del ground_truth_theta

    plt.scatter(alphas, results)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
