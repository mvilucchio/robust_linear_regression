import numpy as np
import numerical_function_double as numfuneps


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
    eps=0.1,
):

    alphas = np.logspace(
        np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), n_alpha_points
    )
    error_theory = np.zeros(n_alpha_points)

    for i, alpha in enumerate(alphas):
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
