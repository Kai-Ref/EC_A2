from ioh import get_problem, ProblemClass
from ioh import logger
import sys
import numpy as np
from tqdm import tqdm
from aco import ACO


# Please replace this `random search` by your `genetic algorithm`.
def mmas(func, budget=None, evaporation_rate=0.02):
    # budget of each run: 50n^2
    if budget is None:
        budget = int(func.meta_data.n_variables * func.meta_data.n_variables * 50)
    if func.meta_data.problem_id == 18 and func.meta_data.n_variables == 32:
        optimum = 8
    else:
        optimum = func.optimum.y
    print(f"The optimal value is {optimum}")
    # 10 independent runs for each algorithm on each problem.
    for _ in range(10):
        aco = ACO(number_bits=func.meta_data.n_variables, evaporation_rate=evaporation_rate)
        f_opt = sys.float_info.min
        x_opt = None
        for _ in tqdm(range(budget)):
            x = aco()
            f = func(x)
            if f >= f_opt or x_opt is None:
                f_opt = f
                x_opt = x
            if f_opt >= optimum:
                break
            aco.update_pheromone(x_opt, f_opt)
        print(f"Best found solution: f={f_opt}, x={x_opt}")
        func.reset()
    return f_opt, x_opt


# Declaration of problems to be tested.
om = get_problem(fid=1, dimension=50, instance=1, problem_class=ProblemClass.PBO)
lo = get_problem(fid=2, dimension=50, instance=1, problem_class=ProblemClass.PBO)
labs = get_problem(fid=18, dimension=50, instance=1, problem_class=ProblemClass.PBO)
# Create default logger compatible with IOHanalyzer
# `root` indicates where the output files are stored.
# `folder_name` is the name of the folder containing all output. You should
# compress this folder and upload it to IOHanalyzer
l = logger.Analyzer(
    root="data",
    folder_name="run",
    algorithm_name="mmas",
    algorithm_info="test of IOHexperimenter in python",
)
om.attach_logger(l)
mmas(om)
lo.attach_logger(l)
mmas(lo)
labs.attach_logger(l)
mmas(labs)
# This statemenet is necessary in case data is not flushed yet.
del l
