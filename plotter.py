# library imports
import os
import matplotlib.pyplot as plt

# project imports


class Plotter:
    """
    This class responsible to plot all the needed graphs for paper
    """

    # CONSTS #

    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def baseline(model_matrix,
                 save_path: str = "baseline.pdf"):
        # TODO: plot the run itself
        ax = plt.subplot(111)

        # plt.plot()
        # plt.plot()
        # plt.plot()
        plt.xlabel("Time", fontsize=14)
        plt.ylabel("Population size", fontsize=14)
        plt.legend()
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.savefig(save_path, dpi=600)
        plt.close()

    @staticmethod
    def sensitivity(x,
                    y,
                    y_err,
                    x_label: str,
                    y_label: str,
                    save_path: str):
        # TODO: plot the run itself
        ax = plt.subplot(111)
        plt.errorbar(x,
                     y,
                     y_err,
                     ecolor="blue",
                     color="blue",
                     capsize=3,
                     fmt="-o")
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.savefig(save_path, dpi=600)
        plt.close()

