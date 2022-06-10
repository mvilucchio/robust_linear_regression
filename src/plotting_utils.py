import os
import datetime as dt
import matplotlib.pyplot as plt

IMG_DIRECTORY = "/Volumes/LaCie/final_imgs_hproblem"  #  "./imgs" #  #  # "/Volumes/LaCie/final_imgs_hproblem" #
STYLES_DIRECTORY = "./src/mpl_styles"


def initialization_mpl(stylesheet_name="latex_ready"):
    plt.style.use(os.path.join(STYLES_DIRECTORY, stylesheet_name + ".mplstyle"))


def save_plot(fig, name, formats=["png", "pdf", "svg"], date=True):
    now = dt.datetime.now()
    for f in formats:
        fig.savefig(
            os.path.join(
                IMG_DIRECTORY,
                "{}".format(name) + "_" + now.strftime("%Y_%m_%d_%H_%M_%S") + "." + f,
            ),
            format=f,
        )


def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
