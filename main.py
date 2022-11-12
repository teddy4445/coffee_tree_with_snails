# library imports
import os
import glob
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# project imports


class Main:
    """
    The main class of the project
    """

    def __init__(self):
        pass

    @staticmethod
    def run() -> None:
        """
        A single entry point to the script, running the entire logic
        :return:
        """
        pass

    @staticmethod
    def io() -> None:
        # TODO: add later 
        for name in []:
            try:
                os.mkdir(os.path.join(os.path.dirname(__file__), name))
            except:
                pass

    @staticmethod
    def solve_the_model() -> tuple:

        vdp1 = lambda T, Y: [Y[1], (1 - Y[0]**2) * Y[1] - Y[0]]
        sol = solve_ivp (vdp1, [0, 20], [2, 0])
        T = sol.t
        Y = sol.y
        return T, Y


if __name__ == '__main__':
    Main.run()
