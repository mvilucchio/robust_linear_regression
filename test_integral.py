import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
from tqdm.auto import tqdm
import fixed_point_equations as fpe
from mpl_toolkits.mplot3d import Axes3D
import numerical_functions as num
from matplotlib import cm


if __name__ == "__main__":
    delta = 0.1

    m, q, sigma = 0.0, 0.0, 0.0

    while True:
        m = np.random.random()
        q = np.random.random()
        sigma = np.random.random()
        if np.square(m) < q + delta * q:
            break

    print("m : {:.3f} q : {:.3f} sigma : {:.3f}".format(m, q, sigma))

    borders = num.find_integration_borders(
        lambda y, zeta : num.integral2(y, zeta, q, m, sigma, delta), 
        0.1, np.sqrt((1 + delta)),
        0.1
    )

    print("border : [{:.3f}, {:.3f}] border 2 : [{:.3f}, {:.3f}]".format(borders[0][0], borders[0][1], borders[1][0], borders[1][1]))

    X = np.linspace(borders[0][0], borders[0][1], 500) # y
    Y = np.linspace(borders[1][0], borders[1][1], 500) # zeta
    X, Y = np.meshgrid(X, Y)
    
    Z = np.empty(shape=(len(X), len(X)))

    for i in range(len(X)):
        for j in range(len(X)):
            Z[i][j] = num.integral2(X[i][j], Y[i][j], q, m, sigma, delta)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlabel('y')
    ax.set_ylabel('zeta')
    # ax.set_zlabel(r'$\Sigma_{final}$')

    # ax.set_title("FP Stability")
    # ax.set_ylabel(r"")
    # ax.set_xlabel(r"$\Sigma_{init}$")
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.minorticks_on()
    ax.grid(True, which='both')
    # ax.legend()
    
    plt.show()