import numpy as np
import os
from re import search
import src.numerics as num
import src.fpeqs as fpe
from src.optimal_lambda import (
    optimal_lambda,
    optimal_reg_param_and_huber_parameter,
)


DATA_FOLDER_PATH = "./data"

FOLDER_PATHS = [
    "./data/experiments",
    "./data/theory",
    "./data/bayes_optimal",
    "./data/reg_param_optimal",
    "./data/reg_param_optimal_experimental",
    "./data/reg_and_huber_param_optimal",
    "./data/reg_and_huber_param_optimal_experimental",
    "./data/others",
]

REG_EXPS = [
    "(exp)",
    "(theory)",
    "(BO|Bayes[ ]{0,1}Optimal)",
    "((reg[\_\s]{0,1}param|lambda)[\_\s]{0,1}optimal)",
    "((reg[\_\s]{0,1}param|lambda)[\_\s]{0,1}optimal)[\_\s]{1}exp",
    "((reg[\_\s]{0,1}param|lambda)[\s]{1}(huber[\_\s]{0,1}param)[\_\s]{0,1}optimal)",
    "((reg[\_\s]{0,1}param|lambda)[\s]{1}(huber[\_\s]{0,1}param)[\_\s]{0,1}optimal)[\_\s]{1}exp",
]

LOSS_NAMES = ["L2", "L1", "Huber"]

SINGLE_NOISE_NAMES = [
    "{loss_name} single noise - exp - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - dim {n_features:d} - rep {repetitions:d} - delta {delta} - lambda {reg_param}",
    "{loss_name} single noise - theory - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta {delta} - lambda {reg_param}",
    "BO single noise - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta {delta}",
    "{loss_name} single noise - reg_param optimal - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta {delta}",
    "{loss_name} single noise - reg_param optimal experimental - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta {delta}",
    "Huber single noise - reg_param and huber_param optimal - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta {delta}",
    "Huber single noise - reg_param and huber_param optimal experimental - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta {delta}",
    "{loss_name} single noise - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta {delta} - lambda {reg_param}",
]

DOUBLE_NOISE_NAMES = [
    "{loss_name} double noise - eps {percentage} - exp - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - dim {n_features:d} - rep {repetitions:d} - delta [{delta_small} {delta_large}] - lambda {reg_param}",
    "{loss_name} double noise - eps {percentage} - theory - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}] - lambda {reg_param}",
    "BO double noise - eps {percentage} - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}]",
    "{loss_name} double noise - eps {percentage} - reg_param optimal - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}]",
    "{loss_name} double noise - eps {percentage} - reg_param optimal experimental - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}]",
    "Huber double noise - eps {percentage} - reg_param and huber_param optimal - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}]",
    "Huber double noise - eps {percentage} - reg_param and huber_param optimal experimental - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}]",
    "{loss_name} double noise - eps {percentage} - alphas [{alpha_min} {alpha_max} {alpha_pts:d}] - delta [{delta_small} {delta_large}] - lambda {reg_param}",
]

# ------------


def _exp_type_choser(test_string, values=[0, 1, 2, 3, 4, 5, 6, -1]):
    for idx, re in enumerate(REG_EXPS):
        if search(re, test_string):
            return values[idx]
    return values[-1]


def _loss_type_chose(test_string, values=[0, 1, 2, 3, 4, 5, 6, -1]):
    for idx, re in enumerate(LOSS_NAMES):
        if search(re, test_string):
            return values[idx]
    return values[-1]


def file_name_generator(**kwargs):
    experiment_code = _exp_type_choser(kwargs["experiment_type"])

    if not (experiment_code == 2 or experiment_code == 5 or experiment_code == 6):
        if kwargs["loss_name"] == "Huber":
            kwargs["loss_name"] += " " + str(kwargs.get("a", 1.0))

    if float(kwargs.get("percentage", 0.0)) == 0.0:
        return SINGLE_NOISE_NAMES[experiment_code].format(**kwargs)
    else:
        return DOUBLE_NOISE_NAMES[experiment_code].format(**kwargs)


def create_check_folders():
    if not os.path.exists(DATA_FOLDER_PATH):
        os.makedirs(DATA_FOLDER_PATH)
        for folder_path in FOLDER_PATHS:
            os.makedirs(folder_path)

    for folder_path in FOLDER_PATHS:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


def check_saved(**kwargs):
    # create_check_folders()

    experiment_code = _exp_type_choser(kwargs["experiment_type"])
    folder_path = FOLDER_PATHS[experiment_code]

    file_path = os.path.join(folder_path, file_name_generator(**kwargs))

    file_exists = os.path.exists(file_path + ".npz")

    return file_exists, (file_path + ".npz")


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
    elif experiment_code == 4:
        np.savez(
            file_path,
            alphas=kwargs["alphas"],
            errors_mean=kwargs["errors_mean"],
            errors_std=kwargs["errors_std"],
            lambdas=kwargs["lambdas"],
        )
    elif experiment_code == 5:
        np.savez(
            file_path,
            alphas=kwargs["alphas"],
            errors=kwargs["errors"],
            lambdas=kwargs["lambdas"],
            huber_params=kwargs["huber_params"],
        )
    elif experiment_code == 6:
        np.savez(
            file_path,
            alphas=kwargs["alphas"],
            errors_mean=kwargs["errors_mean"],
            errors_std=kwargs["errors_std"],
            lambdas=kwargs["lambdas"],
            huber_params=kwargs["huber_params"],
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
        errors = saved_data["errors"]
        lambdas = saved_data["lambdas"]
        return alphas, errors, lambdas
    elif experiment_code == 4:
        alphas = saved_data["alphas"]
        errors_mean = saved_data["errors_mean"]
        errors_std = saved_data["errors_std"]
        lambdas = saved_data["lambdas"]
        return alphas, errors_mean, errors_std, lambdas
    elif experiment_code == 5:
        alphas = saved_data["alphas"]
        errors = saved_data["errors"]
        lambdas = saved_data["lambdas"]
        huber_params = saved_data["huber_params"]
        return alphas, errors, lambdas, huber_params
    elif experiment_code == 6:
        alphas = saved_data["alphas"]
        errors_mean = saved_data["errors_mean"]
        errors_std = saved_data["errors_std"]
        lambdas = saved_data["lambdas"]
        huber_params = saved_data["huber_params"]
        return alphas, errors_mean, errors_std, lambdas, huber_params
    else:
        raise ValueError("experiment_type not recognized.")


# ------------


def experiment_runner(**kwargs):
    experiment_code = _exp_type_choser(kwargs["experiment_type"])

    if experiment_code == 0:
        experimental_points_runner(**kwargs)
    elif experiment_code == 1:
        theory_curve_runner(**kwargs)
    elif experiment_code == 2:
        bayes_optimal_runner(**kwargs)
    elif experiment_code == 3:
        reg_param_optimal_runner(**kwargs)
    elif experiment_code == 4:
        reg_param_optimal_experiment_runner(**kwargs)
    elif experiment_code == 5:
        reg_param_and_huber_param_optimal_runner(**kwargs)
    elif experiment_code == 6:
        reg_param_and_huber_param_experimental_optimal_runner(**kwargs)
    else:
        raise ValueError("experiment_type not recognized.")


def experimental_points_runner(**kwargs):
    _, file_path = check_saved(**kwargs)

    double_noise = not float(kwargs.get("percentage", 0.0)) == 0.0

    if double_noise:
        noise_fun_kwargs = {
            "delta_small": kwargs["delta_small"],
            "delta_large": kwargs["delta_large"],
            "percentage": kwargs["percentage"],
        }
    else:
        noise_fun_kwargs = {"delta": kwargs["delta"]}

    if kwargs["loss_name"] == "Huber":
        find_coefficients_fun_kwargs = {"a": kwargs["a"]}
    else:
        find_coefficients_fun_kwargs = {}

    alphas, errors_mean, errors_std = num.generate_different_alpha(
        num.noise_gen_double if double_noise else num.noise_gen_single,
        _loss_type_chose(
            kwargs["loss_name"],
            values=[
                num.find_coefficients_L2,
                -1,  # num.find_coefficients_L1,
                num.find_coefficients_Huber,
                -1,
            ],
        ),
        alpha_1=kwargs["alpha_min"],
        alpha_2=kwargs["alpha_max"],
        n_features=kwargs["n_features"],
        n_alpha_points=kwargs["alpha_pts"],
        repetitions=kwargs["repetitions"],
        reg_param=kwargs["reg_param"],
        noise_fun_kwargs=noise_fun_kwargs,
        find_coefficients_fun_kwargs=find_coefficients_fun_kwargs,
    )

    kwargs.update(
        {
            "file_path": file_path,
            "alphas": alphas,
            "errors_mean": errors_mean,
            "errors_std": errors_std,
        }
    )

    save_file(**kwargs)


def theory_curve_runner(**kwargs):
    _, file_path = check_saved(**kwargs)

    double_noise = not float(kwargs.get("percentage", 0.0)) == 0.0

    if double_noise:
        var_hat_kwargs = {
            "delta_small": kwargs["delta_small"],
            "delta_large": kwargs["delta_large"],
            "percentage": kwargs["percentage"],
        }

        delta_small = kwargs["delta_small"]
        delta_large = kwargs["delta_large"]

        while True:
            m = 0.89 * np.random.random() + 0.1
            q = 0.89 * np.random.random() + 0.1
            sigma = 0.89 * np.random.random() + 0.1
            if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
                initial_condition = [m, q, sigma]
                break

        var_functions = [
            fpe.var_hat_func_L2_double_noise,
            -1,
            fpe.var_hat_func_Huber_num_double_noise,
            -1,
        ]
    else:
        var_hat_kwargs = {"delta": kwargs["delta"]}

        delta = kwargs["delta"]

        while True:
            m = 0.89 * np.random.random() + 0.1
            q = 0.89 * np.random.random() + 0.1
            sigma = 0.89 * np.random.random() + 0.1
            if np.square(m) < q + delta * q:
                initial_condition = [m, q, sigma]
                break

        var_functions = [
            fpe.var_hat_func_L2_num_single_noise,
            -1,
            fpe.var_hat_func_Huber_num_single_noise,
            -1,
        ]

    if _loss_type_chose(kwargs["loss_name"], values=[False, False, True, False]):
        var_hat_kwargs.update({"a": kwargs["a"]})

    alphas, [errors] = fpe.different_alpha_observables_fpeqs(
        fpe.var_func_L2,
        _loss_type_chose(kwargs["loss_name"], values=var_functions),
        alpha_1=kwargs["alpha_min"],
        alpha_2=kwargs["alpha_max"],
        n_alpha_points=kwargs["alpha_pts"],
        reg_param=kwargs["reg_param"],
        initial_cond=initial_condition,
        var_hat_kwargs=var_hat_kwargs,
    )

    kwargs.update(
        {"file_path": file_path, "alphas": alphas, "errors": errors,}
    )

    save_file(**kwargs)


def bayes_optimal_runner(**kwargs):
    _, file_path = check_saved(**kwargs)

    double_noise = not float(kwargs.get("percentage", 0.0)) == 0.0

    if double_noise:
        var_hat_kwargs = {
            "delta_small": kwargs["delta_small"],
            "delta_large": kwargs["delta_large"],
            "percentage": kwargs["percentage"],
        }

        delta_small = kwargs["delta_small"]
        delta_large = kwargs["delta_large"]

        while True:
            m = 0.89 * np.random.random() + 0.1
            q = 0.89 * np.random.random() + 0.1
            sigma = 0.89 * np.random.random() + 0.1
            if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
                initial_condition = [m, q, sigma]
                break

        var_function = fpe.var_hat_func_BO_double_noise
    else:
        var_hat_kwargs = {"delta": kwargs["delta"]}

        delta = kwargs["delta"]

        while True:
            m = 0.89 * np.random.random() + 0.1
            q = 0.89 * np.random.random() + 0.1
            sigma = 0.89 * np.random.random() + 0.1
            if np.square(m) < q + delta * q:
                initial_condition = [m, q, sigma]
                break

        var_function = fpe.var_hat_func_BO_single_noise

    alphas, (errors, ) = fpe.different_alpha_observables_fpeqs(
        fpe.var_func_BO,
        var_function,
        alpha_1=kwargs["alpha_min"],
        alpha_2=kwargs["alpha_max"],
        n_alpha_points=kwargs["alpha_pts"],
        initial_cond=initial_condition,
        var_hat_kwargs=var_hat_kwargs,
    )

    kwargs.update(
        {"file_path": file_path, "alphas": alphas, "errors": errors,}
    )

    save_file(**kwargs)


def reg_param_optimal_runner(**kwargs):
    _, file_path = check_saved(**kwargs)

    double_noise = not float(kwargs.get("percentage", 0.0)) == 0.0
    
    if double_noise:
        var_hat_kwargs = {
            "delta_small": kwargs["delta_small"],
            "delta_large": kwargs["delta_large"],
            "percentage": kwargs["percentage"],
        }

        delta_small = kwargs["delta_small"]
        delta_large = kwargs["delta_large"]

        while True:
            m = 0.89 * np.random.random() + 0.1
            q = 0.89 * np.random.random() + 0.1
            sigma = 0.89 * np.random.random() + 0.1
            if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
                initial_condition = [m, q, sigma]
                break

        var_functions = [
            fpe.var_hat_func_L2_double_noise,
            -1,
            fpe.var_hat_func_Huber_num_double_noise,
            -1,
        ]
    else:
        var_hat_kwargs = {"delta": kwargs["delta"]}
        delta = kwargs["delta"]

        while True:
            m = 0.89 * np.random.random() + 0.1
            q = 0.89 * np.random.random() + 0.1
            sigma = 0.89 * np.random.random() + 0.1
            if np.square(m) < q + delta * q:
                initial_condition = [m, q, sigma]
                break

        var_functions = [
            fpe.var_hat_func_L2_single_noise,
            -1,
            fpe.var_hat_func_Huber_num_single_noise,
            -1,
        ]

    if _loss_type_chose(kwargs["loss_name"], values=[False, False, True, False]):
        var_hat_kwargs.update({"a": kwargs["a"]})

    alphas, errors, lambdas = optimal_lambda(
        fpe.var_func_L2,
        _loss_type_chose(kwargs["loss_name"], values=var_functions),
        alpha_1=kwargs["alpha_min"],
        alpha_2=kwargs["alpha_max"],
        n_alpha_points=kwargs["alpha_pts"],
        initial_cond=initial_condition,
        var_hat_kwargs=var_hat_kwargs,
    )

    kwargs.update(
        {"file_path": file_path, "alphas": alphas, "errors": errors, "lambdas": lambdas,}
    )

    save_file(**kwargs)


def reg_param_optimal_experiment_runner(**kwargs):
    raise NotImplementedError


def reg_param_and_huber_param_optimal_runner(**kwargs):
    _, file_path = check_saved(**kwargs)

    double_noise = not float(kwargs.get("percentage", 0.0)) == 0.0

    if double_noise:
        var_hat_kwargs = {
            "delta_small": kwargs["delta_small"],
            "delta_large": kwargs["delta_large"],
            "percentage": kwargs["percentage"],
        }

        delta_small = kwargs["delta_small"]
        delta_large = kwargs["delta_large"]

        while True:
            m = 0.89 * np.random.random() + 0.1
            q = 0.89 * np.random.random() + 0.1
            sigma = 0.89 * np.random.random() + 0.1
            if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
                initial_condition = [m, q, sigma]
                break

    else:
        var_hat_kwargs = {"delta": kwargs["delta"]}
        delta = kwargs["delta"]

        while True:
            m = 0.89 * np.random.random() + 0.1
            q = 0.89 * np.random.random() + 0.1
            sigma = 0.89 * np.random.random() + 0.1
            if np.square(m) < q + delta * q:
                initial_condition = [m, q, sigma]
                break

    alphas, errors, lambdas, huber_params = optimal_reg_param_and_huber_parameter(
        double_noise=double_noise,
        alpha_1=kwargs["alpha_min"],
        alpha_2=kwargs["alpha_max"],
        n_alpha_points=kwargs["alpha_pts"],
        initial_cond=initial_condition,
        var_hat_kwargs=var_hat_kwargs,
    )

    kwargs.update(
        {
            "file_path": file_path,
            "alphas": alphas,
            "errors": errors,
            "lambdas": lambdas,
            "huber_params": huber_params,
        }
    )

    save_file(**kwargs)


def reg_param_and_huber_param_experimental_optimal_runner(**kwargs):
    raise NotImplementedError
