import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
from tqdm.auto import tqdm
import fixed_point_equations as fpe
import numerics as num

random_number = np.random.randint(0, 100)

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

alpha_min, alpha_max = 0.01, 100
alpha_points_theory = 75
alpha_points_num = 15
d = 400
reps = 10
deltas = [1.0]
lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]

colormap = get_cmap(len(lambdas) * len(deltas))

alphas_num = [None] * len(deltas) * len(lambdas)
final_errors_mean = [None] * len(deltas) * len(lambdas)
final_errors_std = [None] * len(deltas) * len(lambdas)

for idx, l in enumerate(tqdm(lambdas, desc="lambda", leave=False)):
    for jdx, delta in enumerate(tqdm(deltas, desc="delta", leave=False)):
        i = idx * len(deltas) + jdx
        alphas_num[i], final_errors_mean[i], final_errors_std[i] = num.generate_different_alpha(
            num.find_coefficients_ridge, 
            delta = delta, 
            alpha_1 = alpha_min, 
            alpha_2 = alpha_max, 
            n_features = d, 
            n_alpha_points = alpha_points_num, 
            repetitions = reps, 
            lambda_reg = l
        )

alphas_theory = [None] * len(deltas) * len(lambdas)
errors = [None] * len(deltas) * len(lambdas)

for idx, l in enumerate(lambdas):
    for jdx, delta in enumerate(deltas):
        i = idx * len(deltas) + jdx

        while True:
            m = np.random.random() # 0.1 * np.random.random() 
            q = np.random.random() # 0.3 * np.random.random() + 0.1 # np.random.random()
            sigma = np.random.random() # 0.1 * np.random.random() + 0.05 # np.random.random()
            if np.square(m) < q + delta * q:
                break
            
        initial = [m, q, sigma]

        alphas_theory[i], errors[i] = fpe.projection_ridge_different_alpha_theory(
            fpe.var_func_L2, 
            fpe.var_hat_func_L2, 
            alpha_1 = alpha_min, 
            alpha_2 = alpha_max, 
            n_alpha_points = alpha_points_theory, 
            lambd = l, 
            delta = delta,
            initial_cond = initial
        )

fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)

for idx, l in enumerate(lambdas):
    for jdx, delta in enumerate(deltas):
        i = idx * len(deltas) + jdx
        ax.plot(
            alphas_theory[i], 
            errors[i],
            # marker='.',
            label = r"$\lambda = {}$ $\Delta = {}$".format(l, delta),
            color = colormap(i)
        )

        ax.errorbar(
            alphas_num[i], 
            final_errors_mean[i], 
            final_errors_std[i],
            marker = '.', 
            linestyle = 'None', 
            # label=r"$\lambda = {}$ $\Delta = {}$".format(l, delta),
            color = colormap(i)
        )

ax.set_title("L2 Loss")
ax.set_ylabel(r"$\frac{1}{d} E[||\hat{w} - w^\star||^2]$")
ax.set_xlabel(r"$\alpha$")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([0.009, 110])
ax.minorticks_on()
ax.grid(True, which='both')
ax.legend()

fig.savefig("./imgs/{} - [{:.3f}, {:.3f}, {:.3f}].png".format(random_number, *initial), format='png')

plt.show()