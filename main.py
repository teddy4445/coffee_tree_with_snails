# library imports
import os
import glob
import oct2py
import numpy as np
import pandas as pd
import scipy.io as sio

# project imports

# initialize for the system
from plotter import Plotter

oct2py.octave.addpath(os.path.dirname(__file__))


class Main:
    """
    The main class of the project
    """

    # CONSTS #
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
        # baseline graph - just run the model and plot it
        Plotter.baseline(model_matrix=Main.solve_the_model(),
                         save_path="baseline.pdf")
        # sensitivity graph (1) - pick param 1
        # TODO: good luck
        # sensitivity graph (2) - pick param 2
        # TODO: good luck
        # sensitivity graph (3) - pick param 3
        # TODO: good luck
        # sensitivity graph (4) - pick param 4
        # TODO: good luck

    @staticmethod
    def io() -> None:
        # TODO: add later if needed or remove
        for name in []:
            try:
                os.mkdir(os.path.join(os.path.dirname(__file__), name))
            except:
                pass

    @staticmethod
    def solve_the_model(tspan: list = None,
                        initial_condition: list = None,
                        a: float = 0.025,
                        beta: float = 0.085,
                        k: float = 0.05,
                        gamma: float = 0.05,
                        b: float = 0.025,
                        d: float = 0.15):
        # fix default params
        if tspan is None:
            tspan = [0, 25]
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
        with open(os.path.join(os.path.dirname(__file__), Main.OCTAVE_BASED_SCRIPT_NAME)) as m_source_file:
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
        with open(os.path.join(os.path.dirname(__file__), Main.OCTAVE_RUN_SCRIPT_NAME)) as m_run_file:
            m_run_file.write(script)
        # run the script
        oct2py.octave.run(os.path.join(os.path.dirname(__file__), Main.OCTAVE_RUN_SCRIPT_NAME))
        # load the result file
        return sio.loadmat(os.path.join(os.path.dirname(__file__), Main.OCTAVE_RUN_RESULT_NAME))


if __name__ == '__main__':
    Main.run()
