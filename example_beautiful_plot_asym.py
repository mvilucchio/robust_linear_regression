import matplotlib.pyplot as plt
import plotting_utils as pu
import numpy as np
from asymptotics import plateau_H, plateau_L1

save = True
experimental_points = True
width = 1.0 * 458.63788
# width = 398.3386
random_number = np.random.randint(100)

pu.initialization_mpl()
tuple_size = pu.set_size(width, fraction=0.49)

fig, ax = plt.subplots(1, 1, figsize=tuple_size)  # , tight_layout=True,
# fig.subplots_adjust(left=0.15)
# fig.subplots_adjust(bottom=0.0)
# fig.subplots_adjust(right=0.99)
fig.subplots_adjust(left=0.2)
fig.subplots_adjust(bottom=0.2)
fig.subplots_adjust(top=0.99)
fig.subplots_adjust(right=0.96)


# all you want with the plot
D_IN = 1.
D_OUT = 5.
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

plt.xlabel(r'$a$')
plt.xlim([0.01,100])
plt.ylabel(r'$x$')


if save:
    pu.save_plot(
        fig,
        "large_alpha_different_a_with_deltain_{:.2f}_deltaout_{:.2f}_epsilon_{:.2f}".format(
            D_IN, D_OUT, epsilon
        ),
    )

plt.show()
