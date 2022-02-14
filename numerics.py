import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
from tqdm.auto import tqdm

def noise_gen(n_samples=1000, delta=1):
    error_sample = np.sqrt(delta) * np.random.normal(loc=0.0, scale=1.0, size=(n_samples,))
    return error_sample

def data_generation_single_noise(
        n_features=100, 
        n_samples=1000, 
        n_generalization=200, 
        delta=1
    ):
    theta_0_teacher = np.random.normal(loc=0.0, scale=1.0, size=(n_features,))

    # training data
    xs = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    total_error = noise_gen(n_samples=n_samples, delta=delta)

    # generalizzation data
    xs_gen = np.random.normal(loc=0.0, scale=1.0, size=(n_generalization, n_features))
    # total_error_gen = noise_gen(n_samples=n_generalization, delta=delta)

    ys = np.divide( xs @ theta_0_teacher, np.sqrt(n_features) ) + total_error
    ys_gen = np.divide( xs_gen @ theta_0_teacher, np.sqrt(n_features) ) # + total_error_gen # should not be here

    return xs, ys, xs_gen, ys_gen, theta_0_teacher

def generate_different_alpha(
        find_coefficients_fun, 
        delta = 1.0, 
        alpha_1 = 1, 
        alpha_2 = 1000, 
        n_features = 100, 
        n_alpha_points = 16, 
        repetitions = 10, 
        lambda_reg = 1.0
    ):
    
    alphas = np.logspace(np.log(alpha_1)/np.log(10), np.log(alpha_2)/np.log(10), n_alpha_points)

    final_errors_mean = np.empty((n_alpha_points,))
    final_errors_std = np.empty((n_alpha_points,))

    for jdx, alpha in enumerate(alphas):
        all_gen_errors = np.empty((repetitions,))

        for idx in range(repetitions):
            xs, ys, _, _, ground_truth_theta = data_generation_single_noise(
                n_features = n_features, 
                n_samples = max(int(np.around(n_features * alpha)), 1),
                n_generalization = 1,
                delta = delta
            )

            estimated_theta = find_coefficients_fun(ys, xs, l = lambda_reg)
            
            all_gen_errors[idx] = np.divide(
                np.sum(np.square(
                    ground_truth_theta - estimated_theta
                )),
                n_features
            )

            del xs
            del ys
            del ground_truth_theta

        final_errors_mean[jdx] = np.mean(all_gen_errors)
        final_errors_std[jdx] = np.std(all_gen_errors)

        del all_gen_errors

    return alphas, final_errors_mean, final_errors_std

def find_coefficients_ridge(ys, xs, l = 1.0):
    n, d = xs.shape
    a = np.divide(xs.T.dot(xs), d) + l * np.identity(d)
    b = np.divide(xs.T.dot(ys), np.sqrt(d))
    return np.linalg.solve(a, b)

def find_coefficients_L1(ys, xs, l = 1.0, n_max = 10000, learning_rate = 0.001):
    n, d = xs.shape
    w = np.random.normal(loc=0.0, scale=1.0, size=(d,))
    xs_norm = np.divide(xs, np.sqrt(d))
    for _ in range(n_max):
        grad = - np.sign(ys - xs_norm @ w) @ (xs_norm) + l * w
        w -= learning_rate * grad
    return w

def find_coefficients_Huber(ys, xs, l = 1.0, n_max = 150, learning_rate = 0.01):
    n, d = xs.shape
    w = np.random.normal(loc=0.0, scale=1.0, size=(d,))
    xs_norm = np.divide(xs, np.sqrt(d))
    for _ in range(n_max):
        grad = 0.0
        w -= learning_rate * grad
    return w

if __name__ == "__main__":
    loss = "L1"
    alpha_min, alpha_max = 0.01, 100
    alpha_points = 21
    d = 400
    reps = 10
    deltas = [1.0]
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]

    alphas = [None] * len(deltas) * len(lambdas)
    final_errors_mean = [None] * len(deltas) * len(lambdas)
    final_errors_std = [None] * len(deltas) * len(lambdas)

    for idx, l in enumerate(tqdm(lambdas, desc="lambda", leave=False)):
        for jdx, delta in enumerate(tqdm(deltas, desc="delta", leave=False)):
            i = idx * len(deltas) +  jdx

            if loss == "L2":
                alphas[i], final_errors_mean[i], final_errors_std[i] = generate_different_alpha(
                    find_coefficients_ridge, 
                    delta = delta, 
                    alpha_1 = alpha_min, alpha_2 = alpha_max, 
                    n_features = d, 
                    n_alpha_points = alpha_points, 
                    repetitions = reps, 
                    lambda_reg = l
                )
            elif loss == "L1":
                alphas[i], final_errors_mean[i], final_errors_std[i] = generate_different_alpha(
                    find_coefficients_L1, 
                    delta = delta, 
                    alpha_1 = alpha_min, alpha_2 = alpha_max, 
                    n_features = d, 
                    n_alpha_points = alpha_points, 
                    repetitions = reps, 
                    lambda_reg = l
                )
            elif loss == "Huber":
                alphas[i], final_errors_mean[i], final_errors_std[i] = generate_different_alpha(
                    find_coefficients_L1, 
                    delta = delta, 
                    alpha_1 = alpha_min, alpha_2 = alpha_max, 
                    n_features = d, 
                    n_alpha_points = alpha_points, 
                    repetitions = reps, 
                    lambda_reg = l
                )

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)

    for idx, l in enumerate(lambdas):
        for jdx, delta in enumerate(deltas):
            i = idx * len(deltas) + jdx
            ax.errorbar(
                alphas[i],
                final_errors_mean[i],
                final_errors_std[i],
                marker='.',
                linestyle='solid',
                label=r"$\lambda = {}$ $\Delta = {}$".format(l, delta)
            )
    
    ax.set_title("{} Loss Numerical".format(loss))
    ax.set_ylabel(r"$\frac{1}{d} E[||\hat{w} - w^\star||^2]$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.grid(True, which='both')
    ax.legend()

    fig.savefig("./imgs/L1_exp_n_10000_lr_1e-3.png", format='png')

    plt.show()