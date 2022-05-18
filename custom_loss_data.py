import numpy as np
from src.fpeqs import (
    different_alpha_observables_fpeqs,
    var_func_L2,
    var_hat_func_numerical_loss_single_noise,
)
from src.numerical_functions import precompute_proximals_loss_double_quad_grid

if __name__ == "__main__":

    delta_small = 0.1
    delta_large = 10.0
    percentage = 0.1
    beta = 0.5
    width = 0.5

    var_hat_kwargs = {
        "delta": delta_small,
        "precompute_proximal_func": precompute_proximals_loss_double_quad_grid,
        "loss_args": {"width": width},
    }

    while True:
        m = 0.89 * np.random.random() + 0.1
        q = 0.89 * np.random.random() + 0.1
        sigma = 0.89 * np.random.random() + 0.1
        if np.square(m) < q + delta_small * q and np.square(m) < q + delta_large * q:
            initial_condition = [m, q, sigma]
            break

    alphas, [m, q, sigma] = different_alpha_observables_fpeqs(
        var_func_L2,
        var_hat_func_numerical_loss_single_noise,
        funs=[lambda m, q, sigma: m, lambda m, q, sigma: q, lambda m, q, sigma: sigma],
        alpha_1=0.01,
        alpha_2=100,
        n_alpha_points=36,
        reg_param=0.1,
        initial_cond=initial_condition,
        var_hat_kwargs=var_hat_kwargs,
    )

    np.savez(
        "custom_loss_quad_width_{:.2f}_ds_{}_dl_{}_eps_{}_beta_{}".format(
            width, delta_small, delta_large, percentage, beta
        ),
        alphas=alphas,
        m=m,
        q=q,
        sigma=sigma,
    )
