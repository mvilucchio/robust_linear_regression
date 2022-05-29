from tqdm import tqdm
from src.utils import experiment_runner
import sys
from itertools import product
from tqdm.auto import tqdm

if __name__ == "__main__":
    if len(sys.argv) == 1:
        percentage, delta_small, delta_large = 0.3, 0.1, 5.0
    else:
        percentage, delta_small, delta_large = map(float, sys.argv[1:])

    # reg_params = [0.01, 0.1, 1.0, 10.0, 100.0]
    delta_large = [0.5, 1.0, 2.0, 5.0, 10.0]  # 0.5, 1.0, 2.0, 5.0, 10.0
    betas = [0.0]  # , 0.5, 1.0
    b = betas[0]

    experiments_settings = [
        {
            "loss_name": "L1",
            "alpha_min": 0.01,
            "alpha_max": 1000,
            "alpha_pts": 46,
            # "alpha_pts_theoretical": 16,
            # "alpha_pts_experimental": 6,
            # "delta": 0.5,
            # "reg_param": rp,
            "delta_small": delta_small,
            "delta_large": dl,
            "percentage": percentage,
            # "a": 1.0,
            # "n_features": 500,
            # "repetitions": 4,
            "beta": b,
            "experiment_type": "reg_param optimal",
        }
        for dl in delta_large  # reg_params
    ]

    for exp_dict in tqdm(experiments_settings):
        experiment_runner(**exp_dict)
