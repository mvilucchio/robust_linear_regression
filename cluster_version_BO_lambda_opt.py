import numpy as np
from cluster_version_optimal_lambda import optimal_lambda
import cluster_version_fixed_point_equations_double as fixedpoint
from src.utils import check_saved, save_file, load_file
import sys

alpha_min, alpha_max = 0.01, 100
alpha_points = 21

epsilon, delta_small, delta_large = map(float, sys.argv[1:])

experiment_dict = {
    "alpha_min": alpha_min,
    "alpha_max": alpha_max,
    "alpha_pts": alpha_points,
    "delta_small": delta_small,
    "delta_large": delta_large,
    "epsilon": epsilon,
    "experiment_type": "BO",
}

file_exists, file_path = check_saved(**experiment_dict)

if not file_exists:
    while True:
        m = 0.89 * np.random.random() + 0.1
        q = 0.89 * np.random.random() + 0.1
        sigma = 0.89 * np.random.random() + 0.1
        if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
            break

    initial = [m, q, sigma]

    # the for to parallelize is the one inside this function
    alphas, errors = fixedpoint.projection_ridge_different_alpha_theory(
        fixedpoint.var_func_BO,
        fixedpoint.var_hat_func_BO_num_eps,
        alpha_1=alpha_min,
        alpha_2=alpha_max,
        n_alpha_points=alpha_points,
        lambd=1.0,
        delta_small=delta_small,
        delta_large=delta_large,
        initial_cond=initial,
        verbose=True,
        eps=epsilon,
    )

    experiment_dict.update({"file_path": file_path, "alphas": alphas, "errors": errors})

    save_file(**experiment_dict)
else:
    experiment_dict.update({"file_path": file_path})

    alphas, errors = load_file(**experiment_dict)
