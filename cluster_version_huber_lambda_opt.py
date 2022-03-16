import numpy as np
from cluster_version_optimal_lambda import optimal_lambda
import cluster_version_fixed_point_equations_double as fixedpoint
from src.utils import check_saved, save_file, load_file
import sys
from mpi4py import MPI

alpha_min, alpha_max = 0.01, 100

if len(sys.argv) == 1:
    epsilon, delta_small, delta_large = 0.1, 0.1, 2.0
else:
    epsilon, delta_small, delta_large = map(float, sys.argv[1:])

comm = MPI.COMM_WORLD

experiment_dict = {
    "loss_name": "Huber",
    "alpha_min": alpha_min,
    "alpha_max": alpha_max,
    "alpha_pts": comm.Get_size(),
    "delta_small": delta_small,
    "delta_large": delta_large,
    "epsilon": epsilon,
    "experiment_type": "reg param optimal",
}

file_exists, file_path = check_saved(**experiment_dict)

if not file_exists:
    while True:
        m = 0.89 * np.random.random() + 0.1
        q = 0.89 * np.random.random() + 0.1
        sigma = np.random.random()
        if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
            break

    initial = [m, q, sigma]

    print(epsilon)
    # the call to this function is the one who should be parallelized
    # it is fond in the file cluster_version_optimal_lambda.py
    alphas, errors, lambdas = optimal_lambda(
        fixedpoint.var_func_L2,
        fixedpoint.var_hat_func_Huber_num_eps,
        alpha_1=alpha_min,
        alpha_2=alpha_max,
        delta_small=delta_small,
        delta_large=delta_large,
        initial_cond=initial,
        eps=epsilon
    )

    experiment_dict.update(
        {"file_path": file_path, "alphas": alphas, "errors": errors, "lambdas": lambdas,}
    )

    save_file(**experiment_dict)
else:
    experiment_dict.update({"file_path": file_path})

    alphas, errors, lambdas = load_file(**experiment_dict)