import sys
from src_cluster.utils_cluster_version import experiment_runner

if __name__ == "__main__":

    if len(sys.argv) == 1:
        percentage, delta_small, delta_large = 0.05, 0.1, 2.0
    else:
        percentage, delta_small, delta_large = map(float, sys.argv[1:])

    loss_name = "Huber"

    experiment_settings = {
        "alpha_min": 0.01,
        "alpha_max": 100,
        "alpha_pts": 4,
        "n_features": 500,
        "percentage": percentage,
        "delta_small": delta_small,
        "delta_large": delta_large,
        "experiment_type": "reg_param huber_param optimal",
    }

    experiment_runner(**experiment_settings)