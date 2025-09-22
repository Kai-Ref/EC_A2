from ioh import get_problem, ProblemClass
from ioh import logger
import sys
import numpy as np
from tqdm import tqdm
from aco import ACO
import argparse
import time


# Please replace this `random search` by your `genetic algorithm`.
def mmas(func, budget=None, evaporation_rate=0.02):
    # budget of each run: 50n^2
    if budget is None:
        budget = int(func.meta_data.n_variables * func.meta_data.n_variables * 50)
    if func.meta_data.problem_id == 18 and func.meta_data.n_variables == 32:
        optimum = 8
    else:
        optimum = func.optimum.y
    # print(f"The optimal value is {optimum}")
    # 10 independent runs for each algorithm on each problem.
    start = time.time()
    average_time_70 = 0
    for _ in tqdm(range(100),desc="Runs"):
        aco = ACO(number_bits=func.meta_data.n_variables, evaporation_rate=evaporation_rate)
        f_opt = 0.0
        time_70 = None
        start_this_run = time.time()
        x_opt = None
        for _ in range(budget):
            x = aco()
            f = func(x)
            if f >= f_opt or x_opt is None:
                f_opt = f
                x_opt = x
            if f_opt >= optimum:
                break
            if f_opt >= 0.7 * optimum and time_70 is None:
                time_70 = time.time() - start_this_run
                average_time_70 += time_70
            aco.update_pheromone(x_opt)
        # print(f"Best found solution: f={f_opt}, x={x_opt}")
        func.reset()
    end = time.time()
    print(f"Total time for 100 runs: {end - start} seconds, average time per run: {(end - start) / 100} seconds, evaporation rate: {evaporation_rate}, average time to reach 70% of optimum: {average_time_70 / 100} seconds")
    return f_opt, x_opt


# Declaration of problems to be tested.
n=100
max_iter=100000
om = get_problem(fid=1, dimension=n, instance=1, problem_class=ProblemClass.PBO)
lo = get_problem(fid=2, dimension=n, instance=1, problem_class=ProblemClass.PBO)
labs = get_problem(fid=18, dimension=n, instance=1, problem_class=ProblemClass.PBO)
# Create default logger compatible with IOHanalyzer
# `root` indicates where the output files are stored.
# `folder_name` is the name of the folder containing all output. You should
# compress this folder and upload it to IOHanalyzer
parser = argparse.ArgumentParser()
parser.add_argument('--evaporation_rate', type=float, default=1, help='Evaporation rate for MMAS')
args = parser.parse_args()
evaporation_rate = args.evaporation_rate

l = logger.Analyzer(
    root="data",
    folder_name=f"mmas_{evaporation_rate}_evap",
    algorithm_name=f"mmas_{evaporation_rate}",
    algorithm_info=f"test of IOHexperimenter in python, evaporation_rate={evaporation_rate}",
)
om.attach_logger(l)
mmas(om,budget=max_iter, evaporation_rate=evaporation_rate)
lo.attach_logger(l)
mmas(lo,budget=max_iter, evaporation_rate=evaporation_rate)
labs.attach_logger(l)
mmas(labs,budget=max_iter, evaporation_rate=evaporation_rate)
# This statemenet is necessary in case data is not flushed yet.
del l
