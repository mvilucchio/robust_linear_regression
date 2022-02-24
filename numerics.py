import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.utils import axis0_safe_slice
from sklearn.utils.extmath import safe_sparse_dot
from tqdm.auto import tqdm
import numba as nb

def noise_gen(n_samples=1000, delta=1):
    error_sample = np.sqrt(delta) * np.random.normal(
        loc=0.0, scale=1.0, size=(n_samples,)
    )
    return error_sample


def noise_gen_double(n_samples=1000, delta_small=1, delta_large=10, eps=0.1):
    choice = np.random.choice([0, 1], p=[1 - eps, eps], size=(n_samples,))
    error_sample = np.empty((n_samples, 2))
    error_sample[:, 0] = np.sqrt(delta_small) * np.random.normal(loc=0.0, scale=1.0, size=(n_samples,))
    error_sample[:, 1] = np.sqrt(delta_large) * np.random.normal(loc=0.0, scale=1.0, size=(n_samples,))
    total_error = np.where(choice, error_sample[:, 1], error_sample[:, 0])
    return total_error


def data_generation_single_noise(
    n_features=100, n_samples=1000, n_generalization=200, delta=1
):
    theta_0_teacher = np.random.normal(loc=0.0, scale=1.0, size=(n_features,))

    # training data
    xs = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    total_error = noise_gen(n_samples=n_samples, delta=delta)

    # generalizzation data
    xs_gen = np.random.normal(loc=0.0, scale=1.0, size=(n_generalization, n_features))
    # total_error_gen = noise_gen(n_samples=n_generalization, delta=delta)

    ys = np.divide(xs @ theta_0_teacher, np.sqrt(n_features)) + total_error
    ys_gen = np.divide(
        xs_gen @ theta_0_teacher, np.sqrt(n_features)
    )  # + total_error_gen # should not be here

    return xs, ys, xs_gen, ys_gen, theta_0_teacher


def data_generation_double_noise(
    n_features=100,
    n_samples=1000,
    n_generalization=200,
    delta_small=1.0,
    delta_large=10.0,
    eps=0.1,
):
    theta_0_teacher = np.random.normal(loc=0.0, scale=1.0, size=(n_features,))

    # training data
    xs = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    total_error = noise_gen_double(
        n_samples=n_samples, delta_small=delta_small, delta_large=delta_large, eps=eps
    )

    # generalizzation data
    xs_gen = np.random.normal(loc=0.0, scale=1.0, size=(n_generalization, n_features))
    # total_error_gen = noise_gen(n_samples=n_generalization, delta=delta)

    ys = np.divide(xs @ theta_0_teacher, np.sqrt(n_features)) + total_error
    ys_gen = np.divide(xs_gen @ theta_0_teacher, np.sqrt(n_features)) # + total_error_gen # should not be here

    return xs, ys, xs_gen, ys_gen, theta_0_teacher


def generate_different_alpha(
    find_coefficients_fun,
    delta=1.0,
    alpha_1=1,
    alpha_2=1000,
    n_features=100,
    n_alpha_points=16,
    repetitions=10,
    lambda_reg=1.0,
):

    alphas = np.logspace(
        np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points
    )

    final_errors_mean = np.empty((n_alpha_points,))
    final_errors_std = np.empty((n_alpha_points,))

    for jdx, alpha in enumerate(tqdm(alphas, desc="alpha", leave=False)):
        all_gen_errors = np.empty((repetitions,))

        for idx in tqdm(range(repetitions), desc="reps", leave=False):
            xs, ys, _, _, ground_truth_theta = data_generation_single_noise(
                n_features=n_features,
                n_samples=max(int(np.around(n_features * alpha)), 1),
                n_generalization=1,
                delta=delta,
            )

            estimated_theta = find_coefficients_fun(ys, xs, reg_param=lambda_reg)

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


def generate_different_alpha_double_noise(
    find_coefficients_fun,
    delta_small=1.0,
    delta_large=10.0,
    alpha_1=0.01,
    alpha_2=100,
    n_features=100,
    n_alpha_points=10,
    repetitions=10,
    lambda_reg=1.0,
    eps=0.1,
):

    alphas = np.logspace(
        np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points
    )

    final_errors_mean = np.empty((n_alpha_points,))
    final_errors_std = np.empty((n_alpha_points,))

    for jdx, alpha in enumerate(alphas):
        all_gen_errors = np.empty((repetitions,))

        for idx in tqdm(range(repetitions), desc="reps", leave=False):
            xs, ys, _, _, ground_truth_theta = data_generation_double_noise(
                n_features=n_features,
                n_samples=max(int(np.around(n_features * alpha)), 1),
                n_generalization=1,
                delta_small=delta_small,
                delta_large=delta_large,
                eps=eps
            )

            estimated_theta = find_coefficients_fun(ys, xs, reg_param=lambda_reg)

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
def find_coefficients_ridge(ys, xs, l=1.0):
    n, d = xs.shape
    a = np.divide(xs.T.dot(xs), d) + l * np.identity(d)
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

    gradient = - safe_sparse_dot(sign_sample, xs_norm)
    gradient += reg_param * w

    return loss, gradient


def find_coefficients_L1(ys, xs, reg_param=1.0, max_iter=15000, tol=1e-6):
    n, d = xs.shape
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

def find_coefficients_Huber(ys, xs, reg_param=1.0, max_iter=15000, tol=1e-6, a=1.0):
    n, d = xs.shape
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

if __name__ == "__main__":
    random_number = np.random.randint(0, 100)

    loss = "L1"
    alpha_min, alpha_max = 0.01, 100
    epsil = 0.1
    alpha_points = 17
    d = 400
    reps = 6
    deltas = [[0.1, 1.0], [1.0, 10.0], [10.0, 100.0]]
    lambdas = [0.01, 0.1]

    alphas = [None] * len(deltas) * len(lambdas)
    final_errors_mean = [None] * len(deltas) * len(lambdas)
    final_errors_std = [None] * len(deltas) * len(lambdas)

    for idx, l in enumerate(tqdm(lambdas, desc="lambda", leave=False)):
        for jdx, (d_small, d_large) in enumerate(
            tqdm(deltas, desc="delta", leave=False)
        ):
            i = idx * len(deltas) + jdx

            if loss == "L2":
                (
                    alphas[i],
                    final_errors_mean[i],
                    final_errors_std[i],
                ) = generate_different_alpha_double_noise(
                    find_coefficients_ridge,
                    delta_small=d_small,
                    delta_large=d_large,
                    alpha_1=alpha_min,
                    alpha_2=alpha_max,
                    n_features=d,
                    n_alpha_points=alpha_points,
                    repetitions=reps,
                    lambda_reg=l,
                    eps=epsil,
                )
            elif loss == "L1":
                (
                    alphas[i],
                    final_errors_mean[i],
                    final_errors_std[i],
                ) = generate_different_alpha(
                    find_coefficients_L1,
                    delta=d_small,
                    alpha_1=alpha_min,
                    alpha_2=alpha_max,
                    n_features=d,
                    n_alpha_points=alpha_points,
                    repetitions=reps,
                    lambda_reg=l,
                )
            elif loss == "Huber":
                (
                    alphas[i],
                    final_errors_mean[i],
                    final_errors_std[i],
                ) = generate_different_alpha(
                    find_coefficients_Huber,
                    delta=d_small,
                    alpha_1=alpha_min,
                    alpha_2=alpha_max,
                    n_features=d,
                    n_alpha_points=alpha_points,
                    repetitions=reps,
                    lambda_reg=l,
                )

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)

    for idx, l in enumerate(lambdas):
        for jdx, delta in enumerate(deltas):
            i = idx * len(deltas) + jdx
            ax.errorbar(
                alphas[i],
                final_errors_mean[i],
                final_errors_std[i],
                marker=".",
                linestyle="solid",
                label=r"$\lambda = {}$ $\Delta = {}$".format(l, delta),
            )

    ax.set_title("{} Loss Numerical".format(loss))
    ax.set_ylabel(r"$\frac{1}{d} E[||\hat{w} - w^\star||^2]$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_on()
    ax.grid(True, which="both")
    ax.legend()

    fig.savefig("./imgs/L2_exp_n_10000_lr_1e-3 - {}.png".format(random_number), format="png")

    plt.show()
