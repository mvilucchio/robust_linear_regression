import numpy as np
from scipy.optimize import minimize
import cluster_version_fixed_point_equations_double as fixedpoint
import pandas as pd
from sys import argv
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
    i = comm.Get_rank()
    pool_size = comm.Get_size()

    alphas = np.logspace(
        np.log(alpha_1) / np.log(10), np.log(alpha_2) / np.log(10), pool_size
    )
    alpha = alphas[i]

    initial = initial_cond

    def error_func(reg_param):
        m, q, _ = fixedpoint.state_equations(
            var_func,
            var_hat_func,
            delta_small=delta_small,
            delta_large=delta_large,
            lambd=reg_param,
            alpha=alpha,
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
    
    if i == 0:
        error_theory = np.zeros(pool_size)
        lambd_opt = np.zeros(pool_size)

        for j in range(1,pool_size):
            error_theory[j] = comm.recv(source = j)
        for j in range(1,pool_size):
            lambd_opt[j] = comm.recv(source = j)

        return alphas, error_theory, lambd_opt


    else:
        comm.send(error_theory_instance, dest = 0)
        comm.send(lambd_opt_instance, dest = 0)


if __name__=="__main__":
    eps = float(argv[1])
    delta_small = float(argv[2])
    delta_large = float(argv[3])

    alphas, error_theory, lambd_opt = optimal_lambda(
        fixedpoint.var_func_L2,
        fixedpoint.var_hat_func_Huber_num_eps,
        alpha_1=0.01,
        alpha_2=100,
        n_alpha_points=16,
        delta_small=delta_small,
        delta_large=delta_large,
        initial_cond=[0.6, 0.0, 0.0],
        eps=eps,
    )

    pd.DataFrame(data={"alpha":alphas, "error":error_theory, "lambda":lambd_opt}).to_csv(f"data_cluster/HuberE{eps}DA{delta_small}DB{delta_large}.csv")