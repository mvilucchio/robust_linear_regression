from src.utils import experiment_runner
from tqdm.auto import tqdm
from itertools import product

if __name__ == "__main__":
    percentage, delta_small, delta_large = 0.3, 0.1, 5.0

    # reg_params = [0.01, 0.1, 1.0, 10.0, 100.0]
    deltas_large = [0.5, 1.0, 2.0, 5.0, 10.0]  # 0.5, 1.0, 2.0, 5.0, 10.0
    betas = [0.0]  # , 0.5, 1.0
    b = betas[0]
    dl = 5.0
    percentages = [0.05, 0.1, 0.3]

    experiments_settings = [
        {
            "loss_name": "L1",
            "alpha_min": 0.05,
            "alpha_max": 10000,
            "alpha_pts": 300,
            # "alpha_pts_theoretical": 16,
            # "alpha_pts_experimental": 6,
            # "delta": 0.5,
            # "reg_param": 0.001,
            "delta_small": delta_small,
            "delta_large": dl,
            "percentage": p,
            # "a": 1.0,
            # "n_features": 500,
            # "repetitions": 4,
            "beta": b,
            "experiment_type": "reg_param optimal",
        }
        for dl, p in product(deltas_large, percentages)  # reg_params for p in percentages
    ]
    
    exp_dict = {
            "loss_name": "L1",
            "alpha_min": 0.05,
            "alpha_max": 10000,
            # "alpha_pts": 300,
            "alpha_pts_theoretical": 300,
            "alpha_pts_experimental": 6,
            # "delta": 0.5,
            # "reg_param": 0.001,
            "delta_small": delta_small,
            "delta_large": dl,
            "percentage": 0.1,
            # "a": 1.0,
            # "n_features": 500,
            # "repetitions": 4,
            "beta": b,
            "experiment_type": "reg_param optimal",
        }
    # for exp_dict in tqdm(experiments_settings):
    experiment_runner(**exp_dict)
