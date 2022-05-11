from src.utils import experiment_runner
from itertools import product

if __name__ == "__main__":

    deltas_large = [0.5, 1.0, 2.0, 5.0, 10.0] # 0.5, 1.0, 2.0, 5.0, 10.0
    percentages = [0.01, 0.05, 0.1, 0.3] # 0.01, 0.05, 0.1, 0.3
    loss_name = "Huber"

    experiment_settings = [
        {
            "loss_name" : loss_name,
            "alpha_min": 0.01,
            "alpha_max": 100,
            "alpha_pts": 36,
            "percentage": p,
            "delta_small": 0.1,
            "delta_large": dl,
            "a": 1.0,
            "experiment_type": "reg_param optimal",
        }
        for dl, p in product(deltas_large, percentages)
    ]

    for dic in experiment_settings:
        experiment_runner(**dic)