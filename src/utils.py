import numpy as np
import os
from re import search

MULT_INTEGRAL = 10
TOL_INT = 1e-6
N_TEST_POINTS = 300

DATA_FOLDER_PATH = "./data"

FOLDER_PATHS = [
    "./data/experiments",
    "./data/theory",
    "./data/bayes optimal",
    "./data/reg_param optimal",
    "./data/others",
]


REG_EXPS = [
    "(exp)",
    "(theory)",
    "(BO|Bayes[ ]{0,1}Optimal)",
    "((reg[\_\s]{0,1}param|lambda)[\_\s]{0,1}optimal)",
]

SINGLE_NOISE_NAMES = [
    "{loss_name} single noise - exp - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - dim {dim:d} - rep {rep:d} - delta {delta_small} - lambda {reg_param}",
    "{loss_name} single noise - theory - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta {delta_small} - lambda {reg_param}",
    "BO single noise - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta {delta_small}",
    "{loss_name} single noise - reg_param optimal - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta {delta_small}",
    "{loss_name} single noise - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta {delta_small} - lambda {reg_param}",
]

DOUBLE_NOISE_NAMES = [
    "{loss_name} double noise - eps {epsilon} - exp - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - dim {dim:d} - rep {rep:d} - delta [{delta_small} {delta_large}] - lambda {reg_param}",
    "{loss_name} double noise - eps {epsilon} - theory - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}] - lambda {reg_param}",
    "BO double noise - eps {epsilon} - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}]",
    "{loss_name} double noise - eps {epsilon} - reg_param optimal - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}]",
    "{loss_name} double noise - eps {epsilon} - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}] - lambda {reg_param}",
]

# ------------


def find_integration_borders_square(
    fun, scale1, scale2, mult=MULT_INTEGRAL, tol=TOL_INT, n_points=N_TEST_POINTS
):
    borders = [[-mult * scale1, mult * scale1], [-mult * scale2, mult * scale2]]

    for idx, ax in enumerate(borders):
        for jdx, border in enumerate(ax):

            while True:
                if idx == 0:
                    max_val = np.max(
                        [
                            fun(borders[idx][jdx], pt)
                            for pt in np.linspace(
                                borders[1 if idx == 0 else 0][0],
                                borders[1 if idx == 0 else 0][1],
                                n_points,
                            )
                        ]
                    )
                else:
                    max_val = np.max(
                        [
                            fun(pt, borders[idx][jdx])
                            for pt in np.linspace(
                                borders[1 if idx == 0 else 0][0],
                                borders[1 if idx == 0 else 0][1],
                                n_points,
                            )
                        ]
                    )
                if max_val > tol:
                    borders[idx][jdx] = borders[idx][jdx] + (
                        -1.0 if jdx == 0 else 1.0
                    ) * (scale1 if idx == 0 else scale2)
                else:
                    break

    for ax in borders:
        ax[0] = -np.max(np.abs(ax))
        ax[1] = np.max(np.abs(ax))

    max_val = np.max([borders[0][1], borders[1][1]])

    borders = [[-max_val, max_val], [-max_val, max_val]]

    return borders


def divide_integration_borders_grid(borders, proportion=0.5, sides_square=3):
    max_range = borders[0][1]
    mid_range = proportion * max_range

    # 1 | 2 | 3
    # 4 | 5 | 6
    # 7 | 8 | 9

    domain_xi = [
        [-mid_range, mid_range],
        [-mid_range, mid_range],
        [mid_range, max_range],
        [mid_range, max_range],
        [mid_range, max_range],
        [-mid_range, mid_range],
        [-max_range, -mid_range],
        [-max_range, -mid_range],
        [-max_range, -mid_range],
    ]

    domain_y = [
        [lambda xi: -mid_range, lambda xi: mid_range],
        [lambda xi: mid_range, lambda xi: max_range],
        [lambda xi: mid_range, lambda xi: max_range],
        [lambda xi: -mid_range, lambda xi: mid_range],
        [lambda xi: -max_range, lambda xi: -mid_range],
        [lambda xi: -max_range, lambda xi: -mid_range],
        [lambda xi: -max_range, lambda xi: -mid_range],
        [lambda xi: -mid_range, lambda xi: mid_range],
        [lambda xi: mid_range, lambda xi: max_range],
    ]

    integral_value = 0.0
    for xi_funs, y_funs in zip(domain_xi, domain_y):
        integral_value += dblquad(
            q_integral_BO_eps,
            xi_funs[0],
            xi_funs[1],
            y_funs[0],
            y_funs[1],
            args=(q, m, sigma, delta_small, delta_large, eps),
        )[0]

    return integral_value
    return


# ------------


def _exp_type_choser(test_string, values=[0, 1, 2, 3, -1]):
    for idx, re in enumerate(REG_EXPS):
        if search(re, test_string):
            return values[idx]
    return values[-1]


def file_name_generator(**kwargs):
    experiment_code = _exp_type_choser(kwargs["experiment_type"])
    if float(kwargs.get("epsilon", 0.0)) == 0.0:
        return SINGLE_NOISE_NAMES[experiment_code].format(**kwargs)
    else:
        return DOUBLE_NOISE_NAMES[experiment_code].format(**kwargs)


def create_check_folders():
    data_dir_exists = os.path.exists(DATA_FOLDER_PATH)
    if not data_dir_exists:
        os.makedirs(DATA_FOLDER_PATH)
        for folder_path in FOLDER_PATHS:
            os.makedirs(folder_path)

    for folder_path in FOLDER_PATHS:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


def check_saved(**kwargs):
    create_check_folders()

    experiment_code = _exp_type_choser(kwargs["experiment_type"])
    folder_path = FOLDER_PATHS[experiment_code]

    file_path = os.path.join(folder_path, file_name_generator(**kwargs))

    file_exists = os.path.exists(file_path + ".npz")

    if file_exists:
        return file_exists, (file_path + ".npz")
    else:
        return file_exists, file_path


def save_file(**kwargs):
    file_path = kwargs.get("file_path")

    experiment_code = _exp_type_choser(kwargs["experiment_type"])

    if file_path is None:
        file_path = os.path.join(
            FOLDER_PATHS[experiment_code], file_name_generator(**kwargs)
        )

    if experiment_code == 0:
        np.savez(
            file_path,
            alphas=kwargs["alphas"],
            errors_mean=kwargs["errors_mean"],
            errors_std=kwargs["errors_std"],
        )
    elif experiment_code == 1 or experiment_code == 2:
        np.savez(file_path, alphas=kwargs["alphas"], errors=kwargs["errors"])
    elif experiment_code == 3:
        np.savez(
            file_path,
            alphas=kwargs["alphas"],
            errors=kwargs["errors"],
            lambdas=kwargs["lambdas"],
        )
    else:
        raise ValueError("experiment_type not recognized.")


def load_file(**kwargs):
    file_path = kwargs.get("file_path")

    experiment_code = _exp_type_choser(kwargs["experiment_type"])

    if file_path is None:
        file_path = os.path.join(
            FOLDER_PATHS[experiment_code], file_name_generator(**kwargs) + ".npz"
        )

    saved_data = np.load(file_path)

    if experiment_code == 0:
        alphas = saved_data["alphas"]
        errors_mean = saved_data["errors_mean"]
        errors_std = saved_data["errors_std"]
        return alphas, errors_mean, errors_std
    elif experiment_code == 1 or experiment_code == 2:
        alphas = saved_data["alphas"]
        errors = saved_data["errors"]
        return alphas, errors
    elif experiment_code == 3:
        alphas = saved_data["alphas"]
        errors = (kwargs["errors"],)
        lambdas = (kwargs["lambdas"],)
        return alphas, errors, lambdas
    else:
        raise ValueError("experiment_type not recognized.")

