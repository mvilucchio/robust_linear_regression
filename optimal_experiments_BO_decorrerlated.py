from src.utils import experiment_runner, load_file, bayes_optimal_runner
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import src.fpeqs as fpe
from src.fpeqs_BO import var_func_BO, var_hat_func_BO_num_double_noise,var_hat_func_BO_num_decorrelated_noise



percentage, delta_small, delta_large = 0.3, 0.1, 5.0

deltas_large = [0.5, 1.0, 5.0, 10.0]
b = 0.0
percentages = [0.05, 0.1, 0.3]
dl = 5.0
p = 0.3

experiments_settings = {
        "alpha_min": 0.01,
        "alpha_max": 1000,
        "alpha_pts": 46,
        "delta_small": delta_small,
        "delta_large": dl,
        "percentage": p,
        "beta": 0.0,
        "experiment_type": "BO",
    }

bayes_optimal_runner(**experiments_settings)