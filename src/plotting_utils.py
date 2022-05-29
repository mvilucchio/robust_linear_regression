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

