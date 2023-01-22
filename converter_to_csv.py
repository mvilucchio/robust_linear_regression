from src.utils import npz_to_csv_converter

if __name__ == "__main__":
    percentages = [0.05, 0.1, 0.3]
    deltas_large = [0.5, 1.0, 2.0, 5.0, 10.0]
    betas = [0.0, 0.5, 1.0]

    index_dl = 3
    index_beta = 0
    index_per = 2
    dl = deltas_large[index_dl]
    beta = betas[index_beta]
    p = percentages[index_per]

    dictionaries_to_convert = []

    dictionaries_to_convert.extend(
        [
            {
                "loss_name": "L2",
                "alpha_min": 0.01,
                "alpha_max": 100000,
                "alpha_pts": 100,
                "delta_small": 0.1,
                "delta_large": dl,
                "percentage": p,
                "beta": beta,
                "experiment_type": "reg_param optimal",
            }
            for dl in deltas_large
        ]
    )

    dictionaries_to_convert.extend(
        [
            {
                "loss_name": "L1",
                "alpha_min": 0.01,
                "alpha_max": 100000,
                "alpha_pts": 100,
                "delta_small": 0.1,
                "delta_large": dl,
                "percentage": p,
                "beta": beta,
                "experiment_type": "reg_param optimal",
            }
            for dl in deltas_large
        ]
    )

    dictionaries_to_convert.extend(
        [
            {
                "loss_name": "Huber",
                "alpha_min": 0.01,
                "alpha_max": 100000,
                "alpha_pts": 100,
                "delta_small": 0.1,
                "delta_large": dl,
                "percentage": p,
                "beta": beta,
                "experiment_type": "reg_param huber_param optimal",
            }
            for dl in deltas_large
        ]
    )

    dictionaries_to_convert.extend(
        [
            {
                "alpha_min": 0.01,
                "alpha_max": 1000,
                "alpha_pts": 46,
                "delta_small": 0.1,
                "delta_large": dl,
                "percentage": p,
                "beta": beta,
                "experiment_type": "BO",
            }
            for dl in deltas_large
        ]
    )

    for d in dictionaries_to_convert:
        npz_to_csv_converter(**d)
