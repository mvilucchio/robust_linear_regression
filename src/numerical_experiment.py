import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
import numerical_functions as numfun
from tqdm.auto import tqdm

class NumericalExperiment():
    def __init__(self, data_generator):
        self.data_generator = data_generator
        self.results = {} # {reg_param : [alphas, results]}

    def run_experiment(self, alpha_min, alpha_max, n_alpha_pts, save=None, file_name=None):
        self.alphas = np.logspace(np.log(alpha_min) / np.log(10), np.log(alpha_max) / np.log(10), n_alpha_pts)
        self.values_and_std = np.zeros((n_alpha_pts, 2))

        for idx, alpha in enumerate(tqdm(self.alphas, desc="alpha", disable=not self.verbose, leave=False)):
            a = 0

    def plot_results(self):
        fig = plt.figure(figsize=(15,8))
        ax = fig.gca()
        for reg_param, (alphas, results) in self.results.items():
            ax.errorbar(alphas, results[:,0], results[:,1], label=r"$\lambda = {}$".format(reg_param))
        plt.show()
        return fig

    def save_results(self):
        