import numpy as np
import matplotlib.pyplot as plt
from pyparsing import alphas
import src.fpeqs as fpe
import src.numerics as num
import src.plotting_utils as pu
from tqdm.auto import tqdm
from src.utils import check_saved, load_file
from multiprocessing import Pool

deltas_large = [0.5, 1.0, 2.0, 5.0, 10.0]
beta = 0.0
p = 0.3

L2_settings = [
    {
        "loss_name": "L2",
        "alpha_min": 0.01,
        "alpha_max": 1000,
        "alpha_pts": 46,
        "delta_small": 0.1,
        "delta_large": dl,
        "percentage": p,
        "beta": beta,
        "experiment_type": "reg_param optimal",
    }
    for dl in deltas_large
]

Huber_settings = [
    {
        "loss_name": "Huber",
        "alpha_min": 0.01,
        "alpha_max": 1000,
        "alpha_pts": 46,
        "delta_small": 0.1,
        "delta_large": dl,
        "percentage": p,
        "beta": beta,
        "experiment_type": "reg_param huber_param optimal",
    }
    for dl in deltas_large
]

if __name__ == "__main__":

    alphas_h_total = [None] * len(Huber_settings)
    errors_h_total = [None] * len(Huber_settings)
    std_h_total = [None] * len(Huber_settings)

    alphas_l2_total = [None] * len(L2_settings)
    errors_l2_total = [None] * len(L2_settings)
    std_l2_total = [None] * len(L2_settings)

    for idx, (H_d, L2_d) in enumerate(zip(tqdm(Huber_settings), L2_settings)):
        alphas_H, _, reg_param_H, a_H = load_file(**H_d)
        alphas_L2, _, reg_param_L2 = load_file(**L2_d)

        alpha_H_idx = alphas_H <= 200
        alphas_H_new = alphas_H[alpha_H_idx]
        alphas_H_new = alphas_H_new[::3]
        len_h = len(alphas_H_new)

        alpha_L2_idx = alphas_L2 <= 200
        alphas_L2_new = alphas_L2[alpha_L2_idx]
        alphas_L2_new = alphas_L2_new[::3]
        len_L2 = len(alphas_L2_new)

        # m_h = np.empty((len_h))
        # q_h = np.empty((len_h))
        # sigma_h = np.empty((len_h))

        # m_L2 = np.empty((len_L2))
        # q_L2 = np.empty((len_L2))
        # sigma_L2 = np.empty((len_L2))

        errors_mean_h = np.empty_like(alphas_H_new)
        errors_std_h = np.empty_like(alphas_H_new)

        find_coefficients_fun_kwargs = [{"a": a} for _, a in zip(alphas_H_new, a_H[::3])]
        inputs = [
            (
                a,
                num.measure_gen_decorrelated,
                num.find_coefficients_Huber,
                500,
                10,
                {
                    "delta_small": H_d["delta_small"],
                    "delta_large": H_d["delta_large"],
                    "percentage": H_d["percentage"],
                    "beta": H_d["beta"],
                },
                rp,
                fckw,
            )
            for a, rp, fckw in zip(
                alphas_H_new, reg_param_H[::3], find_coefficients_fun_kwargs
            )
        ]

        with Pool() as pool:
            results = pool.starmap(num._find_numerical_mean_std, inputs)

        for jdx, r in enumerate(results):
            errors_mean_h[jdx] = r[0]
            errors_std_h[jdx] = r[1]

        alphas_h_total[idx] = alphas_H_new
        errors_h_total[idx] = errors_mean_h
        std_h_total[idx] = errors_std_h

        np.savez(
            "H_exp_dl_{:.2f}".format(H_d["delta_large"]),
            alphas=alphas_H_new,
            errors_mean=errors_mean_h,
            errors_std=errors_std_h,
        )

        errors_mean_l2 = np.empty_like(alphas_L2_new)
        errors_std_l2 = np.empty_like(alphas_L2_new)

        find_coefficients_fun_kwargs = [{} for _ in zip(alphas_L2_new)]
        inputs = [
            (
                a,
                num.measure_gen_decorrelated,
                num.find_coefficients_L2,
                500,
                10,
                {
                    "delta_small": L2_d["delta_small"],
                    "delta_large": L2_d["delta_large"],
                    "percentage": L2_d["percentage"],
                    "beta": L2_d["beta"],
                },
                rp,
                fckw,
            )
            for a, rp, fckw in zip(
                alphas_L2_new, reg_param_L2[::3], find_coefficients_fun_kwargs
            )
        ]

        with Pool() as pool:
            results = pool.starmap(num._find_numerical_mean_std, inputs)

        for jdx, r in enumerate(results):
            errors_mean_l2[jdx] = r[0]
            errors_std_l2[jdx] = r[1]

        alphas_l2_total[idx] = alphas_L2_new
        errors_l2_total[idx] = errors_mean_l2
        std_l2_total[idx] = errors_std_l2

        np.savez(
            "L2_exp_dl_{:.2f}".format(L2_d["delta_large"]),
            alphas=alphas_L2_new,
            errors_mean=errors_mean_l2,
            errors_std=errors_std_l2,
        )

    for a, e in zip(alphas_h_total, errors_h_total):
        # return alphas, errors_mean, errors_std
        plt.scatter(a, e)

    for a, e in zip(alphas_l2_total, errors_l2_total):
        # return alphas, errors_mean, errors_std
        plt.scatter(a, e)

    plt.xscale("log")
    plt.yscale("log")
    plt.show()
