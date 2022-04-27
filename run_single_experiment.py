from src.utils import experiment_runner
import sys

if __name__ == "__main__":
    if len(sys.argv) == 1:
        percentage, delta_small, delta_large = 0.05, 0.1, 2.0
    else:
        percentage, delta_small, delta_large = map(float, sys.argv[1:])

    experiments_settings = [
        {
            "loss_name": "L2",
            "alpha_min": 0.01,
            "alpha_max": 100,
            "alpha_pts": 36,
            "delta_small": delta_small,
            "delta_large": delta_large,
            "percentage": percentage,
            "experiment_type": "reg_param optimal",
        }
    ]

    for exp_dict in experiments_settings:
        experiment_runner(**exp_dict)
