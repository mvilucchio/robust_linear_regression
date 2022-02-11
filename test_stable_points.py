import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
from tqdm.auto import tqdm
import fixed_point_equations as fpe
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    max_range = 2
    n_iters = 1000

    alpha = 1.0
    delta = 1.0
    lamb = 1.0

    inital_qs = np.empty((n_iters,))
    inital_ms = np.empty((n_iters,))
    inital_sigmas = np.empty((n_iters,))

    i = 0
    while i < n_iters:
        q, m, sigma = max_range * np.random.random(size=(3,))
        if np.square(m) < q + delta * q:
           inital_qs[i] = q
           inital_ms[i] = m
           inital_sigmas[i] = sigma
           i += 1

    final_ms = np.empty((n_iters,))
    final_qs = np.empty((n_iters,))
    final_sigmas = np.empty((n_iters,))

    for idx in range(n_iters):
        final_ms[idx], final_qs[idx], final_sigmas[idx] = fpe.state_equations(
            fpe.var_func_L2, 
            fpe.var_hat_func_L2, 
            delta = delta, 
            lambd = lamb, 
            alpha = alpha, 
            init = (inital_qs[idx], inital_ms[idx], inital_ms[idx])
        )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(final_qs, final_ms, final_sigmas, marker='.')

    ax.set_xlabel(r'$q_{final}$')
    ax.set_ylabel(r'$m_{final}$')
    ax.set_zlabel(r'$\Sigma_{final}$')

    # ax.set_title("FP Stability")
    # ax.set_ylabel(r"")
    # ax.set_xlabel(r"$\Sigma_{init}$")
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.minorticks_on()
    ax.grid(True, which='both')
    # ax.legend()
    
    plt.show()