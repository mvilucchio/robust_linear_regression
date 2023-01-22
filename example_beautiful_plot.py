import matplotlib.pyplot as plt
import plotting_utils as pu

save = False
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


if save:
    pu.save_plot(
        fig,
        "presentation_total_optimal_confronts_fixed_delta_{:.2f}_beta_{:.2f}".format(
            delta_large, beta
        ),
    )

plt.show()