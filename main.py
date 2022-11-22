# library imports
import os
import time
import oct2py
import numpy as np
import pandas as pd

# project imports

# initialize for the system
from plotter import Plotter
from mat_file_loader import MatFileLoader

oct2py.octave.addpath(os.path.dirname(__file__))


class Main:
    """
    The main class of the project
    """

    # CONSTS #
    RESULTS_FOLDER = "results"
    METRIC_SCORES = [0.34, 0.33, 0.33]
    OCTAVE_BASED_SCRIPT_NAME = "model_solver.txt"
    OCTAVE_RUN_SCRIPT_NAME = "model_solver.m"
    OCTAVE_RUN_RESULT_NAME = "model_answer.mat"

    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def run() -> None:
        """
        A single entry point to the script, running the entire logic
        :return:
        """
        # make sure the IO is file
        Main.io()
        # baseline graphs
        # Main.first_plot()
        # one-dim sensitivity graphs
        Main.second_graph()
        # heatmap sensitivity graphs
        Main.third_graph()
        # optimal sub-section graphs
        Main.fourth_graph()

    @staticmethod
    def io() -> None:
        for name in [Main.RESULTS_FOLDER]:
            try:
                os.mkdir(os.path.join(os.path.dirname(__file__), name))
            except:
                pass

    @staticmethod
    def desire_metric(df: dict) -> tuple:
        """
        Teddy: I decide we want as many healthy trees with as little number of infected trees and snails
        """
        m = np.asarray(df["y"])
        score = Main.METRIC_SCORES[0] * m[:, 0] - Main.METRIC_SCORES[1] * m[:, 1] - Main.METRIC_SCORES[2] * m[:, 2]
        return np.mean(score), np.std(score)

    @staticmethod
    def first_plot() -> None:
        # baseline graph - just run the model and plot it
        initial_conditions = [
            [100, 5, 100],
            [0, 100, 5],
            [97, 3, 50],
            [25, 25, 25]
        ]
        for index, initial_condition in enumerate(initial_conditions):
            print("Main.first_plot: baseline for initial condition: {} (#{}/{}-{:.2f}%)".format(initial_condition,
                                                                                                index + 1,
                                                                                                len(initial_conditions),
                                                                                                (index + 1) * 100 / len(
                                                                                                    initial_conditions)))
            Plotter.baseline(model_matrix=Main.solve_the_model(initial_condition=initial_condition),
                             save_path=os.path.join(Main.RESULTS_FOLDER, "baseline_{}.pdf".format(index)))

    @staticmethod
    def second_graph() -> None:
        ans_mean = []
        ans_std = []
        x = [0.05 * i for i in range(5)]
        for index, parm_val in enumerate(x):
            print("Main.second_graph: sens for a={} (#{}/{}-{:.2f}%)".format(parm_val,
                                                                             index + 1,
                                                                             len(x),
                                                                             (index + 1) * 100 / len(x)))
            mean, std = Main.desire_metric(df=Main.solve_the_model(a=parm_val))
            ans_mean.append(mean)
            ans_std.append(std)
        Plotter.sensitivity(x=x,
                            y=ans_mean,
                            y_err=ans_std,
                            x_label="a",
                            y_label="add-later",
                            save_path=os.path.join(Main.RESULTS_FOLDER, "sensitivity_{}.pdf".format("a")))

        # TODO: repeat the above code for beta, k, gamma, b, d

    @staticmethod
    def third_graph() -> None:
        x = [0.05 * i for i in range(8)]  # TODO: replace with better range
        y = [0.05 * i for i in range(8)]  # TODO: replace with better range
        answer = []
        for x_parm_val in x:
            row = []
            for y_parm_val in y:
                mean, std = Main.desire_metric(df=Main.solve_the_model(a=x_parm_val,
                                                                       gamma=y_parm_val))
                row.append(mean)
            answer.append(row)
        df = pd.DataFrame(data=answer,
                          columns=x,
                          index=y)
        Plotter.heatmap(df=df,
                        x_label="a",
                        y_label="$\gamma$",
                        save_path=os.path.join(Main.RESULTS_FOLDER, "heatmap_{}_{}.pdf".format("a", "gamma")))

        # TODO: repeat the above code for beta cross k and b cross d

    @staticmethod
    def fourth_graph() -> None:
        # TODO: think about it later
        pass

    @staticmethod
    def solve_the_model(tspan: list = None,
                        initial_condition: list = None,
                        a: float = 0.025,
                        beta: float = 0.085,
                        k: float = 0.05,
                        gamma: float = 0.05,
                        b: float = 0.005,
                        d: float = 0.10):
        # fix default params
        if tspan is None:
            tspan = [0, 100]
        if initial_condition is None:
            initial_condition = [100 - 3, 3, 50]

        # make sure the inputs are legit
        if not isinstance(tspan, list) or len(tspan) != 2 or tspan[1] <= tspan[0]:
            raise Exception("Main.solve_the_model: tspan should be a 2-val list [a,b] where b>a.")
        if not isinstance(initial_condition, list) or len(initial_condition) != 3:
            raise Exception("Main.solve_the_model: initial_condition should be a 3-val list.")
        if not (a >= 0 and beta >= 0 and k >= 0 and gamma >= 0 and b >= 0 and d >= 0):
            raise Exception("Main.solve_the_model: all parameter values should be non-negative.")

        # load generic script
        with open(os.path.join(os.path.dirname(__file__), Main.OCTAVE_BASED_SCRIPT_NAME), "r") as m_source_file:
            script = m_source_file.read()
        # update the code
        script = script.format(tspan,
                               initial_condition,
                               a,
                               beta,
                               k,
                               gamma,
                               b,
                               d)
        # save the file for run
        with open(os.path.join(os.path.dirname(__file__), Main.OCTAVE_RUN_SCRIPT_NAME), "w") as m_run_file:
            m_run_file.write(script)
        # run the script
        oct2py.octave.run(os.path.join(os.path.dirname(__file__), Main.OCTAVE_RUN_SCRIPT_NAME))
        # wait to make sure the file is written
        time.sleep(3)
        # load the result file
        return MatFileLoader.read(path=os.path.join(os.path.dirname(__file__), Main.OCTAVE_RUN_RESULT_NAME),
                                  delete_in_end=True)


if __name__ == '__main__':
    Main.run()
