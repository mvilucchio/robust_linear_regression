import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
import pandas as pd


def plateau_L1(D_IN, D_OUT, epsilon, x=.01, toll = 1e-8):
    err = np.inf
    while True:
        y_IN = D_IN + epsilon**2 * x**2
        y_OUT = D_OUT + (epsilon*x + 1)**2

        x_next = -1 / ((1-epsilon)*np.sqrt(y_OUT/y_IN) + epsilon )
        err = np.abs(x_next-x)
        x = x_next
        if err < toll:
            return x

def plateau_H(D_IN, D_OUT, epsilon, a, x=.01, toll = 1e-8):
    err = np.inf
    while True:
        y_IN = D_IN + epsilon**2 * x**2
        y_OUT = D_OUT + (epsilon*x + 1)**2

        x_next = -1 / ((1-epsilon)*erf(a/np.sqrt(2*y_IN))/erf(a/np.sqrt(2*y_OUT)) + epsilon )
        err = np.abs(x_next-x)
        x = x_next
        if err < toll:
            return x

def main():
    D_IN = .1
    D_OUT = 10
    epsilon = .1

    a_list = np.logspace(-2, 2, 128)
    x_list = [plateau_H(D_IN, D_OUT, epsilon, a, x=.01, toll = 1e-8) for a in a_list]

    x_L2 = 1
    x_L1 = plateau_L1(D_IN, D_OUT, epsilon, x=.01, toll = 1e-8)

    plt.axhline(np.abs(x_L1), color='red', label="L1")
    plt.axhline(x_L2, color='black', label="L2")
    plt.plot(a_list, np.abs(x_list), label="LH")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('a')
    plt.ylabel('x')
    plt.show()
    return

def main_test_L1():
    df = pd.read_csv("confront_asym_reg_param_2.0_delta_in_0.1_delta_out_10.0_percentage_0.3.csv")

    eps_list = df["# eps"].to_numpy()
    sim_list = df["Egenl1/Egenl22"].to_numpy()

    D_IN = .1
    D_OUT = 10

    x_list = np.array([plateau_L1(D_IN, D_OUT, epsilon, x=.01, toll = 1e-8) for epsilon in eps_list])
    # plt.plot(eps_list, sim_list)
    # plt.plot(eps_list, -x_list*eps_list**2)
    plt.plot(eps_list, sim_list - x_list**2)
    plt.title("Check L1")
    plt.show()
    return

def main_test_H():
    df = pd.read_csv("huber_confront_asym_reg_param_1.0_delta_in_0.1_delta_out_10.0_a_1.0.csv")

    eps_list = df["# eps"].to_numpy()
    sim_list = df["Egenl1/Egenl22"].to_numpy()

    D_IN = .1
    D_OUT = 10
    a = 1

    x_list = np.array([plateau_H(D_IN, D_OUT, epsilon, a, x=.01, toll = 1e-8) for epsilon in eps_list])
    # plt.plot(eps_list, sim_list)
    # plt.plot(eps_list, x_list**2)
    plt.plot(eps_list, sim_list - x_list**2)
    plt.title("Check Huber")
    plt.show()
    return

if __name__=="__main__":
    main()
    main_test_L1()
    main_test_H()