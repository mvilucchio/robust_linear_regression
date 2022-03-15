import numpy as np
from scipy.optimize import minimize
import fixed_point_equations_double as fixedpoint
from mpi4py import MPI


def optimal_lambda(
    var_func,
    var_hat_func,
    alpha_1=0.01,
    alpha_2=100,
    n_alpha_points=16,
    delta_small=1.0,
    delta_large=10.0,
    initial_cond=[0.6, 0.0, 0.0],
    eps=0.1,
):

    comm = MPI.COMM_WORLD
    current_rank = comm.Get_rank()
    pool_size = comm.Get_size()

    alphas = np.logspace(
        np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), pool_size
    )

    initial = initial_cond
    error_theory = np.zeros(n_alpha_points)
    lambd_opt = np.zeros(n_alpha_points)

    def error_func(reg_param):
        m, q, _ = fixedpoint.state_equations(
            var_func,
            var_hat_func,
            delta_small=delta_small,
            delta_large=delta_large,
            lambd=reg_param,
            alpha=alphas[current_rank],
            eps=eps,
            init=initial,
        )
        return 1 + q - 2 * m

    obj = minimize(error_func, x0=1.0, method="Nelder-Mead")
    if obj.success:
        error_theory_instance = obj.fun
        lambd_opt_instance = obj.x
    else:
        raise RuntimeError("Minima could not be found")

    if current_rank == 0:
        error_theory = np.empty(pool_size)
        lambd_opt = np.empty(pool_size)

        error_theory[0] = error_theory_instance
        lambd_opt[0] = lambd_opt_instance

        for j in range(1, pool_size):
            error_theory[j] = comm.recv(source=j)
        for j in range(1, pool_size):
            lambd_opt[j] = comm.recv(source=j)

        return alphas, error_theory, lambd_opt
    else:
        comm.send(error_theory_instance, dest=0)
        comm.send(lambd_opt_instance, dest=0)

    # return alphas, error_theory, lambd_opt
