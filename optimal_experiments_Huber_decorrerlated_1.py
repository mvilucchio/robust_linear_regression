from src.utils import experiment_runner
from itertools import product
from tqdm.auto import tqdm

if __name__ == "__main__":

    deltas_large = [0.5, 1.0, 2.0, 5.0, 10.0]  # 0.5, 1.0, 2.0, 5.0, 10.0
    percentages = [0.3, 0.1, 0.01, 0.05]  # 0.01, 0.05, 0.1, 0.3
    betas = [0.0]  # , 0.5, 1.0
    loss_name = "L2"

    experiment_settings = [
        {
            "loss_name": loss_name,
            "alpha_min": 0.01,
            "alpha_max": 10000,
            "alpha_pts_theoretical": 200,
            "alpha_pts_experimental": 15,
            "n_features": 1000,
            "repetitions": 10,
            "percentage": p,
            "delta_small": 0.1,
            "delta_large": dl,
            "beta": b,
            "experiment_type": "reg_param optimal exp",
        }
        for b, dl, p in product(betas, deltas_large, percentages)
    ]

    for dic in tqdm(experiment_settings):
        experiment_runner(**dic)
