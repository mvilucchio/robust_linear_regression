from src.utils import experiment_runner
from tqdm.auto import tqdm

if __name__ == "__main__":
    percentage, delta_small, delta_large = 0.3, 0.1, 5.0

    deltas_large = [0.5, 1.0, 2.0, 5.0, 10.0]
    betas = [0.0]
    b = betas[0]

    experiments_settings = [
        {
            "loss_name": "Huber",
            "alpha_min": 0.01,
            "alpha_max": 10000,
            "alpha_pts_theoretical": 150,
            "alpha_pts_experimental": 21,
            "delta_small": delta_small,
            "delta_large": dl,
            "percentage": percentage,
            "n_features": 200,
            "repetitions": 10,
            "beta": b,
            "experiment_type": "reg_param huber_param optimal exp",
        }
        for dl in deltas_large
    ]

    for exp_dict in tqdm(experiments_settings):
        experiment_runner(**exp_dict)
