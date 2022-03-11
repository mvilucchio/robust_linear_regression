import numpy as np
from utils import check_saved, save_file, load_file
from tqdm.auto import tqdm


class FixedPointFinder:
    def __init__(
        self, problem, condition=None, verbose=False, save_state=True,
    ):
        self.problem = problem
        self.condition = condition
        self.verbose = verbose
        self.save_state = save_state
        self.all_alphas = {}
        self.all_fixed_points = {}

    def find_fixed_point_curve(
        self, alpha_min, alpha_max, n_alpha_pts, epsilon, deltas, reg_param, save=None
    ):

        is_already_saved = check_saved_theory(
            self.problem.get_loss_name(),
            epsilon,
            alpha_min,
            alpha_max,
            n_alpha_pts,
            deltas[0],
            deltas[1],
            reg_param,
        )

        if not is_already_saved:
            index_tuple = tuple([epsilon, deltas, reg_param])

            self.all_alphas[index_tuple] = np.logspace(
                np.log(alpha_min) / np.log(10),
                np.log(alpha_max) / np.log(10),
                n_alpha_pts,
            )

            self.all_fixed_points[index_tuple] = np.zeros((n_alpha_pts, 3))

            if self.condition is None:
                initial_pts = 0.89 * np.random.random(size=(3,)) + 0.1
            else:
                while True:
                    initial_pts = np.random.random(size=(3,))
                    if self.condition(*initial_pts):
                        break

            for idx, alpha in enumerate(
                tqdm(
                    self.all_alphas[index_tuple],
                    desc="alpha",
                    disable=not self.verbose,
                    leave=False,
                )
            ):
                (
                    self.all_fixed_points[index_tuple][idx][0],
                    self.all_fixed_points[index_tuple][idx][1],
                    self.all_fixed_points[index_tuple][idx][2],
                ) = self.state_equations(
                    self.problem.var_functions,
                    self.problem.var_hat_functions,
                    alpha,
                    initial_pts,
                    deltas,
                    reg_param,
                )

            if save is None and self.save_state or save:
                save_file_theory(
                    self.alphas[i],
                    self.fixed_points,
                    self.problem.get_loss_name(),
                    epsilon,
                    alpha_min,
                    alpha_max,
                    n_alpha_pts,
                    deltas[0],
                    deltas[1],
                    reg_param,
                )

    def state_equation_iteration(
        self,
        var_func,
        var_hat_func,
        alpha,
        init,
        deltas,
        reg_param,
        blend=0.1,
        abs_tol=1e-06,
        rel_tol=1e-03,
    ):
        m, q, sigma = init[0], init[1], init[2]
        err_rel = 1.0
        err_abs = 1.0
        while err_rel > rel_tol and err_abs > abs_tol:
            m_hat, q_hat, sigma_hat = var_hat_func(
                m, q, sigma, alpha, deltas[0], deltas[1]
            )

            temp_m, temp_q, temp_sigma = m, q, sigma

            m, q, sigma = var_func(
                m_hat, q_hat, sigma_hat, alpha, deltas[0], deltas[1], reg_param
            )

            err_abs = np.max(np.abs([temp_m - m, temp_q - q, temp_sigma - sigma]))
            err_rel = np.max(
                np.abs([(temp_m - m) / m, (temp_q - q) / q, (temp_sigma - sigma) / sigma])
            )

            m = blend * m + (1 - blend) * temp_m
            q = blend * q + (1 - blend) * temp_q
            sigma = blend * sigma + (1 - blend) * temp_sigma

        return m, q, sigma
