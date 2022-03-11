import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numerical_function_double as numfuneps
from src.utils import check_saved, save_file, load_file


def get_cmap(n, name="hsv"):
    return plt.cm.get_cmap(name, n)


def state_equations(
    var_func,
    var_hat_func,
    delta_small=0.1,
    delta_large=1.0,
    lambd=0.01,
    alpha=0.5,
    eps=0.1,
    init=(0.5, 0.5, 0.5),
):
    m, q, sigma = init[0], init[1], init[2]
    err = 1.0
    blend = 0.5
    while err > 1e-6:
        m_hat, q_hat, sigma_hat = var_hat_func(
            m, q, sigma, alpha, delta_small, delta_large, eps
        )

        temp_m, temp_q, temp_sigma = m, q, sigma

        m, q, sigma = var_func(
            m_hat, q_hat, sigma_hat, alpha, delta_small, delta_large, lambd
        )

        err = np.max(np.abs([(temp_m - m), (temp_q - q), (temp_sigma - sigma)]))
        # print(
        #     "error : {:.6f} alpha : {:.3f} m : {:.6f} q : {:.6f} sigma : {:.6f}".format(
        #         err, alpha, m, q, sigma
        #     )
        # )

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
    delta_small=1.0,
    delta_large=10.0,
    initial_cond=[0.6, 0.0, 0.0],
    verbose=False,
    eps=0.1,
):

    alphas = np.logspace(
        np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points
    )
    error_theory = np.zeros(n_alpha_points)

    for i, alpha in enumerate(
        tqdm(alphas, desc="alpha", disable=not verbose, leave=False)
    ):
        m, q, _ = state_equations(
            var_func,
            var_hat_func,
            delta_small=delta_small,
            delta_large=delta_large,
            lambd=lambd,
            alpha=alpha,
            eps=eps,
            init=initial_cond,
        )
        error_theory[i] = 1 + q - 2 * m

    return alphas, error_theory


def var_func_BO(m_hat, q_hat, sigma_hat, alpha, delta_small, delta_large, lambd):
    q = q_hat / (1 + q_hat)
    return q, q, 1 - q


def var_hat_func_BO_num_eps(m, q, sigma, alpha, delta_small, delta_large, eps):
    q_hat = alpha * numfuneps.q_hat_equation_BO_eps(
        m, q, sigma, delta_small, delta_large, eps
    )
    return q_hat, q_hat, q_hat


def var_func_L2(m_hat, q_hat, sigma_hat, alpha, delta_small, delta_large, lambd):
    m = m_hat / (sigma_hat + lambd)
    q = (np.square(m_hat) + q_hat) / np.square(sigma_hat + lambd)
    sigma = 1.0 / (sigma_hat + lambd)
    return m, q, sigma


def var_hat_func_L2_num_eps(m, q, sigma, alpha, delta_small, delta_large, eps):
    m_hat = alpha * numfuneps.m_hat_equation_L2_eps(
        m, q, sigma, delta_small, delta_large, eps=eps
    )
    q_hat = alpha * numfuneps.q_hat_equation_L2_eps(
        m, q, sigma, delta_small, delta_large, eps=eps
    )
    sigma_hat = -alpha * numfuneps.sigma_hat_equation_L2_eps(
        m, q, sigma, delta_small, delta_large, eps=eps
    )
    return m_hat, q_hat, sigma_hat


def var_hat_func_Huber_num_eps(m, q, sigma, alpha, delta_small, delta_large, eps, a=1.0):
    m_hat = alpha * numfuneps.integral_fpe(
        numfuneps.m_integral_Huber_eps,
        numfuneps.border_plus_Huber,
        numfuneps.border_minus_Huber,
        numfuneps.test_fun_upper_Huber,
        m,
        q,
        sigma,
        delta_small,
        delta_large,
        eps,
        a,
    )
    q_hat = alpha * numfuneps.integral_fpe(
        numfuneps.q_integral_Huber_eps,
        numfuneps.border_plus_Huber,
        numfuneps.border_minus_Huber,
        numfuneps.test_fun_upper_Huber,
        m,
        q,
        sigma,
        delta_small,
        delta_large,
        eps,
        a,
    )
    sigma_hat = -alpha * numfuneps.integral_fpe(
        numfuneps.sigma_integral_Huber_eps,
        numfuneps.border_plus_Huber,
        numfuneps.border_minus_Huber,
        numfuneps.test_fun_upper_Huber,
        m,
        q,
        sigma,
        delta_small,
        delta_large,
        eps,
        a,
    )
    return m_hat, q_hat, sigma_hat


if __name__ == "__main__":
    loss_name = "BO"
    alpha_min, alpha_max = 0.01, 100
    eps = 0.1
    alpha_points = 15
    deltas = [[0.1, 3.0], [0.5, 1.5], [1.0, 2.0], [1.0, 5.0], [1.0, 10.0]]  #
    lambdas = [0.01]  # , 0.1, 1.0, 10.0, 100.0

    names_cm = ["Purples", "Blues", "Greens", "Oranges", "Greys"]

    random_number = np.random.randint(0, 100)

    alphas = [None] * len(deltas) * len(lambdas)
    errors = [None] * len(deltas) * len(lambdas)

    colormap = get_cmap(len(lambdas) * len(deltas) + 1)

    for idx, l in enumerate(tqdm(lambdas, desc="lambda", leave=False)):
        for jdx, (delta_small, delta_large) in enumerate(
            tqdm(deltas, desc="delta", leave=False)
        ):
            i = idx * len(deltas) + jdx

            experiment_dict = {
                "loss_name": loss_name,
                "alpha_min": alpha_min,
                "alpha_max": alpha_max,
                "alpha_pts": alpha_points,
                "delta_small": delta_small,
                "delta_large": delta_large,
                "epsilon": eps,
                "experiment_type": "BO",
            }

            file_exists, file_path = check_saved(**experiment_dict)

            if not file_exists:
                while True:
                    m = 0.89 * np.random.random() + 0.1
                    q = 0.89 * np.random.random() + 0.1
                    sigma = 0.89 * np.random.random() + 0.1
                    if (
                        np.square(m) < q + delta_small * q
                        and np.square(m) < q + delta_large * q
                    ):
                        break

                initial = [m, q, sigma]

                alphas[i], errors[i] = projection_ridge_different_alpha_theory(
                    var_func_BO,
                    var_hat_func_BO_num_eps,
                    alpha_1=alpha_min,
                    alpha_2=alpha_max,
                    n_alpha_points=alpha_points,
                    lambd=l,
                    delta_small=delta_small,
                    delta_large=delta_large,
                    initial_cond=initial,
                    verbose=True,
                )

                experiment_dict.update(
                    {"file_path": file_path, "alphas": alphas[i], "errors": errors[i],}
                )

                save_file(**experiment_dict)
            else:
                experiment_dict.update({"file_path": file_path})

                alphas[i], errors[i] = load_file(**experiment_dict)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)

    for idx, l in enumerate(lambdas):
        for jdx, delta in enumerate(deltas):
            i = idx * len(deltas) + jdx
            ax.plot(
                alphas[i],
                errors[i],
                marker=".",
                label=r"$\lambda = {}$ $\Delta = {}$".format(l, delta),
                color=colormap(i),
            )

    ax.set_title("{} Loss - Double Noise".format(loss_name))
    ax.set_ylabel(r"$\frac{1}{d} E[||\hat{w} - w^\star||^2]$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_on()
    ax.grid(True, which="both")
    ax.legend()

    fig.savefig(
        "./imgs/{} - double - {:d}.png".format(loss_name, random_number), format="png",
    )

    plt.show()
