# library imports
import os
import numpy as np
import pandas as pd
import seaborn as sns
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
    def baseline(model_matrix: dict,
                 save_path: str = "baseline.pdf"):
        # TODO: plot the run itself
        ax = plt.subplot(111)

        plt.plot(model_matrix["t"],
                 np.asarray(model_matrix["y"])[:, 0],
                 "-",
                 color="green",
                 label="$T_s(t)$")

        plt.plot(model_matrix["t"],
                 np.asarray(model_matrix["y"])[:, 1],
                 "--",
                 color="red",
                 label="$T_i(t)$")

        plt.plot(model_matrix["t"],
                 np.asarray(model_matrix["y"])[:, 2],
                 "-.",
                 color="blue",
                 label="$S(t)$")
        plt.xlabel("Time", fontsize=14)
        plt.ylabel("Population size", fontsize=14)
        plt.xlim((min(model_matrix["t"]), max(model_matrix["t"])))
        plt.ylim((0, np.max(np.asarray(model_matrix["y"]))))
        plt.legend()
        plt.grid(alpha=0.25,
                 color="black")
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

    @staticmethod
    def heatmap(df: pd.DataFrame,
                x_label: str,
                y_label: str,
                save_path: str):
        # TODO: after picking the metric, rethink about the limitions - i.e., the vmin & vmax arguments
        sns.heatmap(df,
                    vmin=0,
                    vmax=1,
                    annot=False,
                    cmap="coolwarm")
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.savefig(save_path, dpi=600)
        plt.close()

