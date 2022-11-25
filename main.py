# library imports
import time
import pickle
import oct2py
import random
from dtreeviz.trees import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# project imports
from plotter import Plotter
from mat_file_loader import MatFileLoader

# initialize for the system

oct2py.octave.addpath(os.path.dirname(__file__))


class Main:
    """
    The main class of the project
    """

    # CONSTS #
    RESULTS_FOLDER = "results"

    OCTAVE_BASED_SCRIPT_NAME = "model_solver.txt"
    OCTAVE_RUN_SCRIPT_NAME = "model_solver.m"
    OCTAVE_RUN_RESULT_NAME = "model_answer.mat"

    RANDOM_STATE = 73  # Sheldon's number

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
        # Main.second_graph()
        # heatmap sensitivity graphs
        # Main.third_graph()
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
        An approximation to the basic reproduction number
        """
        m = np.asarray(df["y"])
        score = [(m[index, 1] - m[index - 1, 1]) / m[index - 1, 1] for index in range(1, len(m[:, 1]))]
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
        """
        Generate all the parameter sensitivity graphs
        """
        Main.sens(parameter_range=[i * 0.025 for i in range(5)], parameter_name="a")

        Main.sens(parameter_range=[i * 0.025 for i in range(5)], parameter_name="beta")

        Main.sens(parameter_range=[i * 0.025 for i in range(5)], parameter_name="k")

        Main.sens(parameter_range=[i * 0.025 for i in range(5)], parameter_name="gamma")

        Main.sens(parameter_range=[i * 0.025 for i in range(5)], parameter_name="b")

        Main.sens(parameter_range=[i * 0.025 for i in range(5)], parameter_name="d")

    @staticmethod
    def sens(parameter_range: list,
             parameter_name: str) -> None:
        """
        This function generates a one-dim sensitivity analysis
        """
        ans_mean = []
        ans_std = []
        for index, parm_val in enumerate(parameter_range):
            print("Main.second_graph: sens for {}={} (#{}/{}-{:.2f}%)".format(parameter_name,
                                                                              parm_val,
                                                                              index + 1,
                                                                              len(parameter_range),
                                                                              (index + 1) * 100 / len(parameter_range)))
            mean, std = Main.desire_metric(df=Main.solve_the_model_wrapper(params={parameter_name: parm_val}))
            ans_mean.append(mean)
            ans_std.append(std)
        Plotter.sensitivity(x=parameter_range,
                            y=ans_mean,
                            y_err=ans_std,
                            x_label=parameter_name,
                            y_label="Average basic reproduction number",
                            save_path=os.path.join(Main.RESULTS_FOLDER, "sensitivity_{}.pdf".format(parameter_name)))

    @staticmethod
    def third_graph() -> None:
        """
        This function responsible to run all the needed heatmap analysis needed for the paper
        """

        Main.heatmap(x=[i * 0.025 for i in range(4)],
                     y=[i * 0.025 for i in range(4)],
                     x_parameter_name="a",
                     y_parameter_name="gamma")

        Main.heatmap(x=[i * 0.025 for i in range(4)],
                     y=[i * 0.025 for i in range(4)],
                     x_parameter_name="beta",
                     y_parameter_name="k")

        Main.heatmap(x=[i * 0.025 for i in range(4)],
                     y=[i * 0.025 for i in range(4)],
                     x_parameter_name="b",
                     y_parameter_name="d")

        Main.heatmap(x=[i * 0.025 for i in range(4)],
                     y=[i * 0.025 for i in range(4)],
                     x_parameter_name="a",
                     y_parameter_name="d")

    @staticmethod
    def heatmap(x: list,
                y: list,
                x_parameter_name: str,
                y_parameter_name: str) -> None:
        """
        This function is responsible to get two parameters and return the heatmap of them
        """
        answer = []
        for i_index, x_parm_val in enumerate(x):
            row = []
            for j_index, y_parm_val in enumerate(y):
                print("Main.third_graph: sens for {}={} X {}={} (#{}/{}-{:.2f}%)".format(x_parameter_name,
                                                                                         x_parm_val,
                                                                                         y_parameter_name,
                                                                                         y_parm_val,
                                                                                         i_index * len(
                                                                                             x) + j_index + 1,
                                                                                         len(x) * len(y),
                                                                                         100 * (i_index * len(
                                                                                             x) + j_index + 1) / (
                                                                                                 len(x) * len(
                                                                                             y))))
                mean, std = Main.desire_metric(df=Main.solve_the_model_wrapper(params={
                    x_parameter_name: x_parm_val,
                    y_parameter_name: y_parm_val
                }))
                row.append(mean)
            answer.append(row)
        df = pd.DataFrame(data=answer,
                          columns=[round(val, 2) for val in x],
                          index=[round(val, 2) for val in y])
        Plotter.heatmap(df=df,
                        x_label=x_parameter_name,
                        y_label=y_parameter_name,
                        save_path=os.path.join(Main.RESULTS_FOLDER, "heatmap_{}_{}.pdf".format(x_parameter_name,
                                                                                               y_parameter_name)))

    @staticmethod
    def fourth_graph() -> None:
        """
        This function finds an explainable machine learning model to explain the size of snails based on initial conditions
        """
        # TODO: think about a larger model that takes the params values as well
        # Optimization process hyper-parameter #
        MAX_ITER = 10
        X_DELTA = 1

        # generate the dataset
        x = []
        y = []
        for i in range(5):
            ts = random.randint(0, 100)
            ti = random.randint(0, 100)
            # find best 's' in range
            xi_1 = round(ti / 2)
            # Iterating until either the tolerance or max iterations is met
            xi_history = []
            for opt_index in range(MAX_ITER):
                fi = Main.desire_metric(df=Main.solve_the_model_wrapper(initial_condition=[ts, ti, xi_1]))[0]
                dfds = (Main.desire_metric(df=Main.solve_the_model_wrapper(initial_condition=[ts, ti, xi_1 + X_DELTA]))[
                            0] -
                        Main.desire_metric(df=Main.solve_the_model_wrapper(initial_condition=[ts, ti, xi_1 - X_DELTA]))[
                            0]) / 2 * X_DELTA
                xi = round(xi_1 - fi / dfds)  # Newton-Raphson equation
                # edge case from biology
                if xi < 0:
                    xi = 0
                print("IC = [ts={}, ti={}] ---> Iter #{}/{} - xi={:.3f}, xi_1={}, fi={}, dfds={}".format(ts,
                                                                                                         ti,
                                                                                                         opt_index + 1,
                                                                                                         MAX_ITER,
                                                                                                         xi,
                                                                                                         xi_1,
                                                                                                         fi,
                                                                                                         dfds))
                if abs(xi - xi_1) <= 2 * X_DELTA or xi in xi_history:
                    xi_1 = xi
                    break
                xi_history.append(xi)
                xi_1 = xi
            x.append([ts, ti])
            y.append(round(xi_1))

        # organize the results
        x = pd.DataFrame(data=x,
                         columns=["T_s", "T_i"])
        y = pd.DataFrame(data=y,
                         columns=["S"])

        # save data for later usage
        over_all_df = x.copy()
        over_all_df["S"] = y
        over_all_df.to_csv(os.path.join(os.path.dirname(__file__),
                                        Main.RESULTS_FOLDER,
                                        "model_train_data.csv"),
                           index=False)

        Plotter.snail_scatter(df=over_all_df,
                              save_path=os.path.join(os.path.dirname(__file__),
                                                     Main.RESULTS_FOLDER,
                                                     "optimal.pdf"))

        # split to train and test
        x_train, y_train, x_test, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=Main.RANDOM_STATE)
        # find a good model
        model_picker = GridSearchCV(estimator=DecisionTreeClassifier(splitter="best"),
                                    param_grid={
                                        "criterion": ["gini", "entropy"],
                                        "max_depth": [3, 4, 5, 6, 7],
                                        "ccp_alpha": [0, 0.01, 0.025],
                                        "random_state": [Main.RANDOM_STATE]
                                    })
        model_picker.fit(x_train, y_train)
        clf = model_picker.best_estimator_

        # test model
        print(
            "Main.fourth_graph, model train's mae: {:.3f}%".format(mean_absolute_error(clf.predict(x_train), y_train)))
        print("Main.fourth_graph, model test's mae: {:.3f}%".format(mean_absolute_error(clf.predict(x_test), y_test)))

        # plot tree
        Plotter.dt(clf=clf,
                   x=x,
                   y=y,
                   feature_names=list(x),
                   save_path=os.path.join(Main.RESULTS_FOLDER, "fig_4.pdf"))

        # save model to file for later if needed
        with open(os.path.join(os.path.dirname(__file__), Main.RESULTS_FOLDER, "model"), "wb") as model_file:
            pickle.dump(clf,
                        model_file)

    @staticmethod
    def solve_the_model_wrapper(params: dict = None,
                                tspan: list = None,
                                initial_condition: list = None):
        """
        A function responsible to let set the model's parameter values by name
        """
        params = {} if params is None else params
        return Main.solve_the_model(tspan=tspan,
                                    initial_condition=initial_condition,
                                    a=0.025 if "a" not in params else params["a"],
                                    beta=0.085 if "beta" not in params else params["beta"],
                                    k=0.05 if "k" not in params else params["k"],
                                    gamma=0.05 if "gamma" not in params else params["gamma"],
                                    d=0.005 if "d" not in params else params["d"],
                                    b=0.10 if "b" not in params else params["b"])

    @staticmethod
    def solve_the_model(tspan: list = None,
                        initial_condition: list = None,
                        a: float = 0.01,
                        beta: float = 0.085,
                        k: float = 0.05,
                        gamma: float = 0.05,
                        b: float = 0.005,
                        d: float = 0.10):
        # fix default params
        if tspan is None:
            tspan = [0, 20]
        if initial_condition is None:
            initial_condition = [100 - 3, 3, 10]

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
