from tqdm import tqdm
from src.utils import experiment_runner
import sys
from tqdm.auto import tqdm

if __name__ == "__main__":
    if len(sys.argv) == 1:
        percentage, delta_small, delta_large = 0.1, 0.1, 5.0
    else:
        percentage, delta_small, delta_large = map(float, sys.argv[1:])

    aa = [0.5, 1.0, 1.5]

    experiments_settings = [
        {
            "loss_name": "L2",
            "alpha_min": 0.01,
            "alpha_max": 100,
            "alpha_pts": 36,
            "reg_param": 1.0,
            "delta" : 0.5,
            "a" : a,
            # "delta_small": delta_small,
            # "delta_large": delta_large,
            # "percentage": percentage,
            # "a": 1.0,
            # "beta": 0.0,
            "experiment_type": "BO",
        }
        for a in aa
    ]

    for exp_dict in tqdm(experiments_settings):
        experiment_runner(**exp_dict)