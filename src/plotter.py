import numpy as np
import matplotlib.pyplot as plt
from utils import file_name_generator, save_file
import os

from tqdm.auto import tqdm

import numerics as num
import fixed_point_equations as fpe


def _get_cmap(n, name="hsv"):
    return plt.cm.get_cmap(name, n)


class Plotter:
    def __init__(
        self,
        cmap_names=["Purples", "Blues", "Greens", "Oranges", "Reds", "Greys",],
        fname_gen_fun=file_name_generator,
        figsize=(15, 8),
        dpi=200,
    ):
        self.cmap_names = cmap_names

        self.fname_gen_fun = fname_gen_fun

        self.fig, self.ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)

        return

    def plot(self, experim):
        if experim:
            self.ax.errorbar()
        else:
            self.ax.plot()

        return

    def get_fig_ax(self):
        return self.fig, self.ax

