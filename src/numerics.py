import numpy as np
from scipy import optimize
from numbers import Number
from sklearn.utils import axis0_safe_slice
from sklearn.utils.extmath import safe_sparse_dot
from numba import njit
import cvxpy as cp

from multiprocessing import Pool

# from mpi4py.futures import MPIPoolExecutor as Pool


def measure_gen_single(generalization, teacher_vector, xs, delta):
    n_samples, n_features = xs.shape
    w_xs = np.divide(xs @ teacher_vector, np.sqrt(n_features))
    if generalization:
        ys = w_xs
    else:
        error_sample = np.sqrt(delta) * np.random.normal(
            loc=0.0, scale=1.0, size=(n_samples,)
        )
        ys = w_xs + error_sample
    return ys


def measure_gen_double(
    generalization, teacher_vector, xs, delta_small, delta_large, percentage
):
    n_samples, n_features = xs.shape
    w_xs = np.divide(xs @ teacher_vector, np.sqrt(n_features))
    if generalization:
        ys = w_xs
    else:
        choice = np.random.choice(
            [0, 1], p=[1 - percentage, percentage], size=(n_samples,)
        )
        error_sample = np.empty((n_samples, 2))
        error_sample[:, 0] = np.sqrt(delta_small) * np.random.normal(
            loc=0.0, scale=1.0, size=(n_samples,)
        )
        error_sample[:, 1] = np.sqrt(delta_large) * np.random.normal(
            loc=0.0, scale=1.0, size=(n_samples,)
        )
        total_error = np.where(choice, error_sample[:, 1], error_sample[:, 0])
        ys = w_xs + total_error
    return ys


def measure_gen_decorrelated(
    generalization, teacher_vector, xs, delta_small, delta_large, percentage, beta
):
    n_samples, n_features = xs.shape
    w_xs = np.divide(xs @ teacher_vector, np.sqrt(n_features))
    if generalization:
        ys = w_xs
    else:
        choice = np.random.choice(
            [0, 1], p=[1 - percentage, percentage], size=(n_samples,)
        )
        error_sample = np.empty((n_samples, 2))
        error_sample[:, 0] = np.sqrt(delta_small) * np.random.normal(
            loc=0.0, scale=1.0, size=(n_samples,)
        )
        error_sample[:, 1] = np.sqrt(delta_large) * np.random.normal(
            loc=0.0, scale=1.0, size=(n_samples,)
        )
        total_error = np.where(choice, error_sample[:, 1], error_sample[:, 0])
        factor_in_front = np.where(choice, beta, 1.0)
        ys = factor_in_front * w_xs + total_error
    return ys


def data_generation(
    measure_fun, n_features, n_samples, n_generalization, measure_fun_kwargs
):
    theta_0_teacher = np.random.normal(loc=0.0, scale=1.0, size=(n_features,))

    xs = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    xs_gen = np.random.normal(loc=0.0, scale=1.0, size=(n_generalization, n_features))

    ys = measure_fun(False, theta_0_teacher, xs, **measure_fun_kwargs)
    ys_gen = measure_fun(True, theta_0_teacher, xs_gen, **measure_fun_kwargs)

    return xs, ys, xs_gen, ys_gen, theta_0_teacher


def _find_numerical_mean_std(
    alpha,
    measure_fun,
    find_coefficients_fun,
    n_features,
    repetitions,
    measure_fun_kwargs,
    reg_param,
    find_coefficients_fun_kwargs,
):
    all_gen_errors = np.empty((repetitions,))

    for idx in range(repetitions):
        xs, ys, _, _, ground_truth_theta = data_generation(
            measure_fun,
            n_features=n_features,
            n_samples=max(int(np.around(n_features * alpha)), 1),
            n_generalization=1,
            measure_fun_kwargs=measure_fun_kwargs,
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

    error_mean, error_std = np.mean(all_gen_errors), np.std(all_gen_errors)

    del all_gen_errors

    return error_mean, error_std


def no_parallel_generate_different_alpha(
    measure_fun,
    find_coefficients_fun,
    alpha_1=0.01,
    alpha_2=100,
    n_features=100,
    n_alpha_points=10,
    repetitions=10,
    reg_param=1.0,
    measure_fun_kwargs={"delta_small": 0.1, "delta_large": 10.0, "percentage": 0.1},
    find_coefficients_fun_kwargs={},
    alphas=None,
):
    if alphas is None:
        alphas = np.logspace(
            np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points
        )
    else:
        n_alpha_points = len(alphas)

    if isinstance(reg_param, Number):
        reg_param = reg_param * np.ones_like(alphas)

    if not isinstance(find_coefficients_fun_kwargs, list):
        find_coefficients_fun_kwargs = [find_coefficients_fun_kwargs] * len(alphas)

    errors_mean = np.empty((n_alpha_points,))
    errors_std = np.empty((n_alpha_points,))

    results = []
    for a, rp, fckw in zip(alphas, reg_param, find_coefficients_fun_kwargs):
        results.append(
            _find_numerical_mean_std(
                a,
                measure_fun,
                find_coefficients_fun,
                n_features,
                repetitions,
                measure_fun_kwargs,
                rp,
                fckw,
            )
        )

    for idx, r in enumerate(results):
        errors_mean[idx] = r[0]
        errors_std[idx] = r[1]

    return alphas, errors_mean, errors_std


def generate_different_alpha(
    measure_fun,
    find_coefficients_fun,
    alpha_1=0.01,
    alpha_2=100,
    n_features=100,
    n_alpha_points=10,
    repetitions=10,
    reg_param=1.0,
    measure_fun_kwargs={"delta_small": 0.1, "delta_large": 10.0, "percentage": 0.1},
    find_coefficients_fun_kwargs={},
    alphas=None,
):
    if alphas is None:
        alphas = np.logspace(
            np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points
        )
    else:
        n_alpha_points = len(alphas)

    if isinstance(reg_param, Number):
        reg_param = reg_param * np.ones_like(alphas)

    if not isinstance(find_coefficients_fun_kwargs, list):
        find_coefficients_fun_kwargs = [find_coefficients_fun_kwargs] * len(alphas)

    errors_mean = np.empty((n_alpha_points,))
    errors_std = np.empty((n_alpha_points,))

    inputs = [
        (
            a,
            measure_fun,
            find_coefficients_fun,
            n_features,
            repetitions,
            measure_fun_kwargs,
            rp,
            fckw,
        )
        for a, rp, fckw in zip(alphas, reg_param, find_coefficients_fun_kwargs)
    ]

    with Pool(processes=1) as pool:
        results = pool.starmap(_find_numerical_mean_std, inputs)

    print("---", len(results))
    for idx, r in enumerate(results):
        errors_mean[idx] = r[0]
        errors_std[idx] = r[1]

    return alphas, errors_mean, errors_std


@njit(error_model="numpy", fastmath=True)
def find_coefficients_L2(ys, xs, reg_param):
    _, d = xs.shape
    a = np.divide(xs.T.dot(xs), d) + reg_param * np.identity(d)
    b = np.divide(xs.T.dot(ys), np.sqrt(d))
    return np.linalg.solve(a, b)


# @njit(error_model="numpy", fastmath=True)
# def _loss_and_gradient_L1(w, xs_norm, ys, reg_param):
#     linear_loss = ys - xs_norm @ w

#     loss = np.sum(np.abs(linear_loss)) + 0.5 * reg_param * np.dot(w, w)

#     sign_sample = np.ones_like(linear_loss)
#     sign_sample_mask = linear_loss < 0
#     zero_sample_mask = linear_loss == 0
#     sign_sample[sign_sample_mask] = -1.0
#     sign_sample[zero_sample_mask] = 0.0

#     gradient = -safe_sparse_dot(sign_sample, xs_norm)
#     gradient += reg_param * w

#     return loss, gradient


# def find_coefficients_L1(ys, xs, reg_param, max_iter=15000, tol=1e-6):
#     _, d = xs.shape
#     w = np.random.normal(loc=0.0, scale=1.0, size=(d,))
#     xs_norm = np.divide(xs, np.sqrt(d))

#     bounds = np.tile([-np.inf, np.inf], (w.shape[0], 1))
#     bounds[-1][0] = np.finfo(np.float64).eps * 10

#     opt_res = optimize.minimize(
#         _loss_and_gradient_L1,
#         w,
#         method="L-BFGS-B",
#         jac=True,
#         args=(xs_norm, ys, reg_param),
#         options={"maxiter": max_iter, "gtol": tol, "iprint": -1},
#         bounds=bounds,
#     )

#     if opt_res.status == 2:
#         raise ValueError(
#             "L1Regressor convergence failed: l-BFGS-b solver terminated with %s"
#             % opt_res.message
#         )

#     return opt_res.x


def find_coefficients_L1(ys, xs, reg_param):
    _, d = xs.shape
    # w = np.random.normal(loc=0.0, scale=1.0, size=(d,))
    xs_norm = np.divide(xs, np.sqrt(d))
    w = cp.Variable(shape=d)
    obj = cp.Minimize(cp.norm(ys - xs_norm @ w, 1) + 0.5 * reg_param * cp.sum_squares(w))
    prob = cp.Problem(obj)
    prob.solve(eps_abs=1e-3)

    return w.value


@njit(error_model="numpy", fastmath=True)
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


@njit(error_model="numpy", fastmath=True)
def _loss_and_gradient_cutted_l2(w, xs_norm, ys, reg_param, a):
    linear_loss = ys - xs_norm @ w
    abs_linear_loss = np.abs(linear_loss)
    outliers_mask = abs_linear_loss > a

    outliers = abs_linear_loss[outliers_mask]
    num_outliers = np.count_nonzero(outliers_mask)
    n_non_outliers = xs_norm.shape[0] - num_outliers

    non_outliers = linear_loss[~outliers_mask]
    loss = 0.5 * np.dot(
        non_outliers, non_outliers
    )  # 0.0  # a * np.sum(outliers) - 0.5 * num_outliers * a ** 2
    loss += 0.5 * reg_param * np.dot(w, w)

    xs_non_outliers = -axis0_safe_slice(xs_norm, ~outliers_mask, n_non_outliers)

    signed_outliers = np.ones_like(outliers)
    signed_outliers_mask = linear_loss[outliers_mask] < 0
    signed_outliers[signed_outliers_mask] = -1.0

    # xs_outliers = axis0_safe_slice(xs_norm, outliers_mask, num_outliers)

    # gradient -= a * safe_sparse_dot(signed_outliers, xs_outliers)

    gradient = safe_sparse_dot(non_outliers, xs_non_outliers)
    gradient += reg_param * w

    return loss, gradient


def find_coefficients_cutted_l2(ys, xs, reg_param, a=1.0, max_iter=150, tol=1e-3):
    _, d = xs.shape
    w = np.random.normal(loc=0.0, scale=1.0, size=(d,))
    xs_norm = np.divide(xs, np.sqrt(d))

    bounds = np.tile([-np.inf, np.inf], (w.shape[0], 1))
    bounds[-1][0] = np.finfo(np.float64).eps * 10

    opt_res = optimize.minimize(
        _loss_and_gradient_cutted_l2,
        w,
        # method="L-BFGS-B",
        jac=True,
        args=(xs_norm, ys, reg_param, a),
        options={"maxiter": max_iter, "gtol": tol},
        # bounds=bounds,
    )

    if opt_res.status == 2:
        raise ValueError(
            "Cutted L2 Regressor convergence failed: solver terminated with %s"
            % opt_res.message
        )

    return opt_res.x
