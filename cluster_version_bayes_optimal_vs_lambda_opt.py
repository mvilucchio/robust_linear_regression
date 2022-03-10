import numpy as np
from cluster_version_optimal_lambda import optimal_lambda
import cluster_version_fixed_point_equations_double as fixedpoint
from src.utils import check_saved

alpha_min, alpha_max = 0.01, 100
eps = [0.01, 0.1, 0.3]
alpha_points = 21
deltas = [
    [1.0, 1.1],
    [1.0, 1.2],
    [1.0, 1.3],
    [1.0, 1.4],
    [1.0, 1.5],
    [0.5, 0.6],
    [0.5, 0.7],
    [0.5, 0.8],
    [0.5, 0.9],
    [0.5, 1.0],
]

alphas = [None] * len(deltas) * len(eps)
errors = [None] * len(deltas) * len(eps)
lambdas = [None] * len(deltas) * len(eps)

for idx, e in enumerate(eps):
    for jdx, (delta_small, delta_large) in enumerate(deltas):
        i = idx * len(deltas) + jdx

        file_exists, file_path = check_saved(
            loss_name="Huber",
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            alpha_pts=alpha_points,
            delta_small=delta_small,
            delta_large=delta_large,
            epsilon=e,
            experiment_type="reg param optimal",
        )

        if not file_exists:
            while True:
                m = 0.89 * np.random.random() + 0.1
                q = 0.89 * np.random.random() + 0.1
                sigma = np.random.random()
                if (
                    np.square(m) < q + delta_small * q
                    and np.square(m) < q + delta_large * q
                ):
                    break

            initial = [m, q, sigma]

            alphas[i], errors[i], lambdas[i] = optimal_lambda(
                fixedpoint.var_func_L2,
                fixedpoint.var_hat_func_Huber_num_eps,
                alpha_1=alpha_min,
                alpha_2=alpha_max,
                n_alpha_points=alpha_points,
                delta_small=delta_small,
                delta_large=delta_large,
                initial_cond=initial,
                eps=e,
                verbose=True,
            )

            np.savez(
                file_path, alphas=alphas[i], errors=errors[i], lambdas=lambdas[i],
            )

        else:
            data = np.load(file_path)

            alphas[i] = data["alphas"]
            errors[i] = data["errors"]
            lambdas[i] = data["lambdas"]

alphas_L2 = [None] * len(deltas) * len(eps)
errors_L2 = [None] * len(deltas) * len(eps)
lambdas_L2 = [None] * len(deltas) * len(eps)

colormap = fixedpoint.get_cmap(len(eps) * len(deltas) + 1)

for idx, e in enumerate(eps):
    for jdx, (delta_small, delta_large) in enumerate(deltas):
        i = idx * len(deltas) + jdx

        file_exists, file_path = check_saved(
            loss_name="L2",
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            alpha_pts=alpha_points,
            delta_small=delta_small,
            delta_large=delta_large,
            epsilon=e,
            experiment_type="reg param optimal",
        )

        if not file_exists:
            while True:
                m = 0.89 * np.random.random() + 0.1
                q = 0.89 * np.random.random() + 0.1
                sigma = np.random.random()
                if (
                    np.square(m) < q + delta_small * q
                    and np.square(m) < q + delta_large * q
                ):
                    break

            initial = [m, q, sigma]

            alphas_L2[i], errors_L2[i], lambdas_L2[i] = optimal_lambda(
                fixedpoint.var_func_L2,
                fixedpoint.var_hat_func_L2_num_eps,
                alpha_1=alpha_min,
                alpha_2=alpha_max,
                n_alpha_points=alpha_points,
                delta_small=delta_small,
                delta_large=delta_large,
                initial_cond=initial,
                eps=e,
                verbose=True,
            )

            np.savez(
                file_path,
                alphas=alphas_L2[i],
                errors=errors_L2[i],
                lambdas=lambdas_L2[i],
            )

        else:
            data = np.load(file_path)

            alphas_L2[i] = data["alphas"]
            errors_L2[i] = data["errors"]
            lambdas_L2[i] = data["lambdas"]
