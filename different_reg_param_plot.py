import numpy as np
import src_cluster.fpeqs_cluster_version as fpes
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker

N=15
step = 0.1
lw = 2.0
delta = 0.1
alphas = np.arange(0.2, 2.1, step)

def myfmt(x, pos):
    return '{0:.1f}'.format(x)

if __name__ == "__main__":
    reg_params = [None] * len(alphas)
    gen_errors = [None] * len(alphas)

    cmap = plt.get_cmap("viridis", len(alphas))

    for idx, a in enumerate(tqdm(alphas)):
        reg_params[idx], (gen_errors[idx],) = fpes.different_reg_param_gen_error(
            fpes.var_func_L2,
            fpes.var_hat_func_L2_single_noise,
            funs=[lambda m, q, sigma: 1 + q - 2 * m],
            reg_param_1=0.01,
            reg_param_2=10,
            n_reg_param_points=300,
            alpha=a,
            initial_cond=[0.8, 0.8, 0.8],
            var_hat_kwargs={"delta" : delta},
        )

    font = {'size': 18, 'family' : 'xyzxyz'}
    mpl.rc('font', **font)
    text = {'usetex' : True, 'latex.preamble': r'\usepackage{amsfonts} \usepackage{physics}'}
    mpl.rc('text', **text)
    legend = {'edgecolor' : 'w', 'handlelength' : 1.4}
    mpl.rc('legend', **legend)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True,)

    norm = mpl.colors.Normalize(vmin=0.2,vmax=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for idx, (rp, ge) in enumerate(zip(reg_params, gen_errors)):
        min_indices = np.where(ge == ge.min())

        ax.plot(
            rp, 
            ge, 
            linewidth=lw,
            # marker=".",
            color=cmap(idx),
            # linestyle='dotted',
            label=r"$\alpha$ = {:.1f}".format(alphas[idx])
        )

        ax.plot(
            rp[min_indices],
            ge[min_indices],
            linestyle=None,
            marker=".",
            color='r'
        )
    

    ax.set_title("$\Delta$ = {:.1f}".format(delta))
    ax.set_ylabel(r"Generalization Error: $\frac{1}{d} \mathbb{E}\qty[\norm{\bf{\hat{w}} - \bf{w^\star}}^2]$")
    ax.set_xlabel(r"Regularization Parameter $\lambda$ ")
    ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.set_xlim([0.009, 110])
    ax.minorticks_on()
    ax.grid(True, which="both")
    # ax.legend()

    cbar = fig.colorbar(sm, ticks=np.arange(0.2,2.1,step), boundaries=np.arange(0.15,2.15,step), format=ticker.FuncFormatter(myfmt))

    cbar.ax.set_title(r"$\alpha$")

    plt.show()