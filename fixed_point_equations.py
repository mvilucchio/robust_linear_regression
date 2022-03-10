import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
from tqdm.auto import tqdm
import numerical_functions as numfun
import numerical_function_double as numfuneps


def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def state_equations(
    var_func, var_hat_func, delta=0.001, lambd=0.01, alpha=0.5, init=(0.5, 0.5, 0.5)
):
    m, q, sigma = init[0], init[1], init[2]
    err = 1.0
    blend = 0.5
    while err > 1e-6:
        m_hat, q_hat, sigma_hat = var_hat_func(m, q, sigma, alpha, delta)

        temp_m, temp_q, temp_sigma = m, q, sigma

        m, q, sigma = var_func(m_hat, q_hat, sigma_hat, alpha, delta, lambd)

        err = np.max(np.abs([(temp_m - m), (temp_q - q), (temp_sigma - sigma)]))

        m = blend * m + (1 - blend) * temp_m
        q = blend * q + (1 - blend) * temp_q
        sigma = blend * sigma + (1 - blend) * temp_sigma

    return m, q, sigma


def projection_ridge_different_alpha_theory(
    var_func,
    var_hat_func,
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    lambd=0.1,
    delta=1.0,
    initial_cond=[0.6, 0.0, 0.0],
    verbose=False,
):

    alphas = np.logspace(
        np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points
    )

    initial = initial_cond
    error_theory = np.zeros(n_alpha_points)

    for i, alpha in enumerate(
        tqdm(alphas, desc="alpha", disable=not verbose, leave=False)
    ):
        m, q, _ = state_equations(
            var_func, var_hat_func, delta, lambd, alpha, init=initial
        )
        error_theory[i] = 1 + q - 2 * m

    return alphas, error_theory


def var_func_BO(m_hat, q_hat, sigma_hat, alpha, delta, lambd):
    q = q_hat / (1 + q_hat)
    return q, q, 1 - q


def var_hat_func_BO(m, q, sigma, alpha, delta):
    q_hat = alpha / (1 + delta - q)
    return q_hat, q_hat, q_hat


def var_hat_func_BO_num(m, q, sigma, alpha, delta):
    q_hat = alpha * numfun.q_hat_equation_BO(m, q, sigma, delta)
    return q_hat, q_hat, q_hat


def var_func_L2(m_hat, q_hat, sigma_hat, alpha, delta, lambd):
    m = m_hat / (sigma_hat + lambd)
    q = (np.square(m_hat) + q_hat) / np.square(sigma_hat + lambd)
    sigma = 1.0 / (sigma_hat + lambd)
    return m, q, sigma


def var_hat_func_L2(m, q, sigma, alpha, delta):
    m_hat = alpha / (1 + sigma)
    q_hat = alpha * (1 + q + delta - 2 * np.abs(m)) / ((1 + sigma) ** 2)
    sigma_hat = alpha / (1 + sigma)
    return m_hat, q_hat, sigma_hat


def var_hat_func_L2_num(m, q, sigma, alpha, delta):
    m_hat = alpha * numfun.m_hat_equation_L2(m, q, sigma, delta)
    q_hat = alpha * numfun.q_hat_equation_L2(m, q, sigma, delta)
    sigma_hat = -alpha * numfun.sigma_hat_equation_L2(m, q, sigma, delta)
    return m_hat, q_hat, sigma_hat


def var_hat_func_L1_num(m, q, sigma, alpha, delta):
    m_hat = alpha * numfun.m_hat_equation_L1(m, q, sigma, delta)
    q_hat = alpha * numfun.q_hat_equation_L1(m, q, sigma, delta)
    sigma_hat = -alpha * numfun.sigma_hat_equation_L1(m, q, sigma, delta)
    return m_hat, q_hat, sigma_hat


def var_hat_func_Huber_num(m, q, sigma, alpha, delta):
    if sigma + 1 >= 0:
        m_hat = alpha * numfun.integral_fpe(
            numfun.m_integral_Huber,
            numfun.border_plus,
            numfun.border_minus,
            numfun.test_fun_upper,
            m,
            q,
            sigma,
            delta,
        )
        q_hat = alpha * numfun.integral_fpe(
            numfun.q_integral_Huber,
            numfun.border_plus,
            numfun.border_minus,
            numfun.test_fun_upper,
            m,
            q,
            sigma,
            delta,
        )
        sigma_hat = -alpha * numfun.integral_fpe(
            numfun.sigma_integral_Huber,
            numfun.border_plus,
            numfun.border_minus,
            numfun.test_fun_upper,
            m,
            q,
            sigma,
            delta,
        )
    else:
        m_hat = alpha * numfun.integral_fpe(
            numfun.m_integral_Huber,
            numfun.border_minus,
            numfun.border_plus,
            numfun.test_fun_down,
            m,
            q,
            sigma,
            delta,
        )
        q_hat = alpha * numfun.integral_fpe(
            numfun.q_integral_Huber,
            numfun.border_minus,
            numfun.border_plus,
            numfun.test_fun_down,
            m,
            q,
            sigma,
            delta,
        )
        sigma_hat = -alpha * numfun.integral_fpe(
            numfun.sigma_integral_Huber,
            numfun.border_minus,
            numfun.border_plus,
            numfun.test_fun_down,
            m,
            q,
            sigma,
            delta,
        )
    return m_hat, q_hat, sigma_hat


def var_hat_func_L2_num_eps(m, q, sigma, alpha, delta_small, delta_large, eps=0.1):
    m_hat = alpha * numfuneps.m_hat_equation_L2_eps(
        m, q, sigma, delta_small, delta_large, eps
    )
    q_hat = alpha * numfuneps.q_hat_equation_L2_eps(
        m, q, sigma, delta_small, delta_large, eps
    )
    sigma_hat = -alpha * numfuneps.sigma_hat_equation_L2_eps(
        m, q, sigma, delta_small, delta_large, eps
    )
    return m_hat, q_hat, sigma_hat


if __name__ == "__main__":
    alpha_min, alpha_max = 0.01, 100
    alpha_points_int = 51
    alpha_points_an = 100
    deltas = [1.0]
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]

    alphas_int = [None] * len(deltas) * len(lambdas)
    errors_int = [None] * len(deltas) * len(lambdas)

    alphas_an = [None] * len(deltas) * len(lambdas)
    errors_an = [None] * len(deltas) * len(lambdas)

    colormap = get_cmap(len(lambdas) * len(deltas) + 1)

    # m = 0.05 * np.random.random() + 0.45
    # q = 0.02 * np.random.random() + 0.01
    # sigma = 0.1 * np.random.random() + 0.05

    for idx, l in enumerate(tqdm(lambdas, desc="lambda", leave=False)):
        for jdx, delta in enumerate(tqdm(deltas, desc="delta", leave=False)):
            while True:
                m = 0.89 * np.random.random() + 0.1
                q = 0.89 * np.random.random() + 0.1
                sigma = 0.89 * np.random.random() + 0.1
                if np.square(m) < q + delta * q:
                    break

            initial = [m, q, sigma]
            print(initial)

            i = idx * len(deltas) + jdx

            alphas_int[i], errors_int[i] = projection_ridge_different_alpha_theory(
                var_func_BO,
                var_hat_func_BO_num,
                alpha_1=alpha_min,
                alpha_2=alpha_max,
                n_alpha_points=alpha_points_int,
                lambd=l,
                delta=delta,
                initial_cond=initial,
                verbose=True,
            )

    # for idx, l in enumerate(tqdm(lambdas, desc="lambda", leave=False)):
    #     for jdx, delta in enumerate(tqdm(deltas, desc="delta", leave=False)):
    #         while True:
    #             m = np.random.random()
    #             q = np.random.random()
    #             sigma = np.random.random()
    #             if np.square(m) < q + delta * q:
    #                 break

    #         initial = [m, q, sigma]

    #         i = idx * len(deltas) + jdx

    #         alphas_an[i], errors_an[i] = projection_ridge_different_alpha_theory(
    #             var_func_L2,
    #             var_hat_func_L2,
    #             alpha_1 = alpha_min,
    #             alpha_2 = alpha_max,
    #             n_alpha_points = alpha_points_an,
    #             lambd = l,
    #             delta = delta,
    #             initial_cond = initial,
    #             verbose = True
    #         )
    #         # print(initial)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)

    for idx, l in enumerate(lambdas):
        for jdx, delta in enumerate(deltas):
            i = idx * len(deltas) + jdx
            ax.plot(
                alphas_int[i],
                errors_int[i],
                marker=".",
                label=r"$\lambda = {}$ $\Delta = {}$".format(l, delta),
                color=colormap(i),
            )
            # ax.plot(
            #     alphas_an[i],
            #     errors_an[i],
            #     # label=r"$\lambda = {}$ $\Delta = {}$".format(l, delta),
            #     color = colormap(i)
            # )

    ax.set_title("Huber Loss")
    ax.set_ylabel(r"$\frac{1}{d} E[||\hat{w} - w^\star||^2]$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_on()
    ax.grid(True, which="both")
    ax.legend()

    fig.savefig(
        "./imgs/hub - [{:.3f}, {:.3f}, {:.3f}].png".format(*initial), format="png"
    )

    plt.show()
