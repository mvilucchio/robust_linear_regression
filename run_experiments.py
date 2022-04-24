from src.utils import experiment_runner

experiments_settings = [
    {
        "alpha_min": 0.01,
        "alpha_max": 100,
        "alpha_pts": 3,
        "reg_param": 0.1,
        "repetitions": 2,
        "n_features": 500,
        "delta": 0.1,
        "experiment_type": "BO",
    },
    {
        "alpha_min": 0.01,
        "alpha_max": 100,
        "alpha_pts": 3,
        "reg_param": 0.1,
        "repetitions": 2,
        "n_features": 500,
        "delta_small": 0.1,
        "delta_large": 10.0,
        "percentage": 0.1,
        "experiment_type": "BO",
    },
]

for i, exp_dict in enumerate(experiments_settings):
    experiment_runner(**exp_dict)
