import numpy as np
from scipy import optimize
from sklearn.utils import axis0_safe_slice
from sklearn.utils.extmath import safe_sparse_dot
from tqdm.auto import tqdm
import numba as nb


def noise_gen_single(n_samples=1000, delta=1):
    error_sample = np.sqrt(delta) * np.random.normal(
        loc=0.0, scale=1.0, size=(n_samples,)
    )
    return error_sample


def noise_gen_double(n_samples=1000, delta_small=1, delta_large=10, percentage=0.1):
    choice = np.random.choice([0, 1], p=[1 - percentage, percentage], size=(n_samples,))
    error_sample = np.empty((n_samples, 2))
    error_sample[:, 0] = np.sqrt(delta_small) * np.random.normal(
        loc=0.0, scale=1.0, size=(n_samples,)
    )
    error_sample[:, 1] = np.sqrt(delta_large) * np.random.normal(
        loc=0.0, scale=1.0, size=(n_samples,)
    )
    total_error = np.where(choice, error_sample[:, 1], error_sample[:, 0])
    return total_error


def data_generation(
    noise_fun,
    n_features=100,
    n_samples=1000,
    n_generalization=200,
    noise_fun_kwargs={"delta_small": 0.1, "delta_large": 2.0, "percentage": 0.1},
):
    theta_0_teacher = np.random.normal(loc=0.0, scale=1.0, size=(n_features,))

    xs = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    total_error = noise_fun(n_samples=n_samples, **noise_fun_kwargs)

    xs_gen = np.random.normal(loc=0.0, scale=1.0, size=(n_generalization, n_features))

    ys = np.divide(xs @ theta_0_teacher, np.sqrt(n_features)) + total_error
    ys_gen = np.divide(xs_gen @ theta_0_teacher, np.sqrt(n_features))

    return xs, ys, xs_gen, ys_gen, theta_0_teacher


def generate_different_alpha(
    noise_fun,
    find_coefficients_fun,
    alpha_1=0.01,
    alpha_2=100,
    n_features=100,
    n_alpha_points=10,
    repetitions=10,
    reg_param=1.0,
    noise_fun_kwargs={"delta_small": 0.1, "delta_large": 10.0, "percentage": 0.1},
    find_coefficients_fun_kwargs={},
):

    alphas = np.logspace(
        np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points
    )

    final_errors_mean = np.empty((n_alpha_points,))
    final_errors_std = np.empty((n_alpha_points,))

    for jdx, alpha in enumerate(tqdm(alphas, desc="alpha", leave=False)):
        all_gen_errors = np.empty((repetitions,))

        for idx in tqdm(range(repetitions), desc="reps", leave=False):
            xs, ys, _, _, ground_truth_theta = data_generation(
                noise_fun,
                n_features=n_features,
                n_samples=max(int(np.around(n_features * alpha)), 1),
                n_generalization=1,
                noise_fun_kwargs=noise_fun_kwargs,
            )

            estimated_theta = find_coefficients_fun(
                ys, xs, reg_param, **find_coefficients_fun_kwargs
            )

            all_gen_errors[idx] = np.divide(
                np.sum(np.square(ground_truth_theta - estimated_theta)), n_features
            )

            del xs
            del ys
            del ground_truth_theta

        final_errors_mean[jdx] = np.mean(all_gen_errors)
        final_errors_std[jdx] = np.std(all_gen_errors)

        del all_gen_errors

    return alphas, final_errors_mean, final_errors_std


@nb.njit(error_model="numpy", fastmath=True)
def find_coefficients_L2(ys, xs, reg_param):
    _, d = xs.shape
    a = np.divide(xs.T.dot(xs), d) + reg_param * np.identity(d)
    b = np.divide(xs.T.dot(ys), np.sqrt(d))
    return np.linalg.solve(a, b)


# @nb.njit(error_model="numpy", fastmath=True)
def _loss_and_gradient_L1(w, xs_norm, ys, reg_param):
    linear_loss = ys - xs_norm @ w

    loss = np.sum(np.abs(linear_loss)) + 0.5 * reg_param * np.dot(w, w)

    sign_sample = np.ones_like(linear_loss)
    sign_sample_mask = linear_loss < 0
    zero_sample_mask = linear_loss == 0
    sign_sample[sign_sample_mask] = -1.0
    sign_sample[zero_sample_mask] = 0.0

    gradient = -safe_sparse_dot(sign_sample, xs_norm)
    gradient += reg_param * w

    return loss, gradient


def find_coefficients_L1(ys, xs, reg_param, max_iter=15000, tol=1e-6):
    _, d = xs.shape
    w = np.random.normal(loc=0.0, scale=1.0, size=(d,))
    xs_norm = np.divide(xs, np.sqrt(d))

    bounds = np.tile([-np.inf, np.inf], (w.shape[0], 1))
    bounds[-1][0] = np.finfo(np.float64).eps * 10

    opt_res = optimize.minimize(
        _loss_and_gradient_L1,
        w,
        method="L-BFGS-B",
        jac=True,
        args=(xs_norm, ys, reg_param),
        options={"maxiter": max_iter, "gtol": tol, "iprint": -1},
        bounds=bounds,
    )

    if opt_res.status == 2:
        raise ValueError(
            "L1Regressor convergence failed: l-BFGS-b solver terminated with %s"
            % opt_res.message
        )

    return opt_res.x


# @nb.njit(error_model="numpy", fastmath=True)
def _loss_and_gradient_huber(w, xs_norm, ys, reg_param, a):
    linear_loss = ys - xs_norm @ w
    abs_linear_loss = np.abs(linear_loss)
    outliers_mask = abs_linear_loss > a

    outliers = abs_linear_loss[outliers_mask]
    num_outliers = np.count_nonzero(outliers_mask)
    n_non_outliers = xs_norm.shape[0] - num_outliers

    loss = a * np.sum(outliers) - 0.5 * num_outliers * a ** 2

    non_outliers = linear_loss[~outliers_mask]
    loss += 0.5 * np.dot(non_outliers, non_outliers)
    loss += 0.5 * reg_param * np.dot(w, w)

    xs_non_outliers = -axis0_safe_slice(xs_norm, ~outliers_mask, n_non_outliers)
    gradient = safe_sparse_dot(non_outliers, xs_non_outliers)

    signed_outliers = np.ones_like(outliers)
    signed_outliers_mask = linear_loss[outliers_mask] < 0
    signed_outliers[signed_outliers_mask] = -1.0

    xs_outliers = axis0_safe_slice(xs_norm, outliers_mask, num_outliers)

    gradient -= a * safe_sparse_dot(signed_outliers, xs_outliers)
    gradient += reg_param * w

    return loss, gradient


def find_coefficients_Huber(ys, xs, reg_param, a=1.0, max_iter=15000, tol=1e-6):
    _, d = xs.shape
    w = np.random.normal(loc=0.0, scale=1.0, size=(d,))
    xs_norm = np.divide(xs, np.sqrt(d))

    bounds = np.tile([-np.inf, np.inf], (w.shape[0], 1))
    bounds[-1][0] = np.finfo(np.float64).eps * 10

    opt_res = optimize.minimize(
        _loss_and_gradient_huber,
        w,
        method="L-BFGS-B",
        jac=True,
        args=(xs_norm, ys, reg_param, a),
        options={"maxiter": max_iter, "gtol": tol, "iprint": -1},
        bounds=bounds,
    )

    if opt_res.status == 2:
        raise ValueError(
            "HuberRegressor convergence failed: l-BFGS-b solver terminated with %s"
            % opt_res.message
        )

    return opt_res.x
