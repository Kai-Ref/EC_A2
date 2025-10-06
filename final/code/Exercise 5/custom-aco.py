from ioh import get_problem, ProblemClass
from ioh import logger
import numpy as np
from tqdm import tqdm
from aco import ACO
import time

evaporation_rate = 0.1
num_ants = 15
num_runs = 10
max_iter = 100000

alpha = 1


# # Please replace this `random search` by your `genetic algorithm`.
def custom_aco(func, budget, evaporation_rate):
    # budget of each run: 50n^2
    if budget is None:
        budget = int(func.meta_data.n_variables * func.meta_data.n_variables * 50)

    if func.meta_data.problem_id == 18 and func.meta_data.n_variables == 32:
        optimum = 8
    else:
        optimum = func.optimum.y
    # print(f"The optimal value is {optimum}")

    start = time.time()
    for _ in tqdm(range(num_runs), desc="Runs"):
        f_opt = -np.inf
        x_opt = None

        aco = ACO(func.meta_data.n_variables, evaporation_rate, num_ants, budget,
                  pheromone_min=1e-6,
                  pheromone_max=100,
                  epsilon=0.01)

        aco.run(func, x_opt, f_opt, optimum)

        func.reset()
    end = time.time()
    print(
        f"\nTotal time for {num_runs} runs: {end - start} seconds, average time per run: {(end - start) / 100} seconds, evaporation rate: {evaporation_rate}")


# Declaration of problems to be tested.
om = get_problem(fid=1, dimension=100, instance=1, problem_class=ProblemClass.PBO)
lo = get_problem(fid=2, dimension=100, instance=1, problem_class=ProblemClass.PBO)
lhw = get_problem(fid=3, dimension=100, instance=1, problem_class=ProblemClass.PBO)
labs = get_problem(fid=18, dimension=100, instance=1, problem_class=ProblemClass.PBO)
nqp = get_problem(fid=23, dimension=100, instance=1, problem_class=ProblemClass.PBO)
ct = get_problem(fid=24, dimension=100, instance=1, problem_class=ProblemClass.PBO)
nkl = get_problem(fid=25, dimension=100, instance=1, problem_class=ProblemClass.PBO)

# Create default logger compatible with IOHanalyzer
# `root` indicates where the output files are stored.
# `folder_name` is the name of the folder containing all output. You should
# compress this folder and upload it to IOHanalyzer

l = logger.Analyzer(root="data", folder_name=f"custom_aco_{evaporation_rate}_evap",
    algorithm_name=f"custom_aco_{evaporation_rate}",
    algorithm_info=f"test of IOHexperimenter in python, evaporation_rate={evaporation_rate}", )

print(f"F{om.meta_data.problem_id}: {om.optimum.y}")
print(f"F{lo.meta_data.problem_id}: {lo.optimum.y}")
print(f"F{lhw.meta_data.problem_id}: {lhw.optimum.y}")
print(f"F{labs.meta_data.problem_id}: {labs.optimum.y}")
print(f"F{nqp.meta_data.problem_id}: {nqp.optimum.y}")
print(f"F{ct.meta_data.problem_id}: {ct.optimum.y}")
print(f"F{nkl.meta_data.problem_id}: {nkl.optimum.y}")

om.attach_logger(l)
custom_aco(om, budget=max_iter, evaporation_rate=evaporation_rate)

lo.attach_logger(l)
custom_aco(lo, budget=max_iter, evaporation_rate=evaporation_rate)

lhw.attach_logger(l)
custom_aco(lhw, budget=max_iter, evaporation_rate=evaporation_rate)

labs.attach_logger(l)
custom_aco(labs, budget=max_iter, evaporation_rate=evaporation_rate)

nqp.attach_logger(l)
custom_aco(nqp, budget=max_iter, evaporation_rate=evaporation_rate)

ct.attach_logger(l)
custom_aco(ct, budget=max_iter, evaporation_rate=evaporation_rate)

nkl.attach_logger(l)
custom_aco(nkl, budget=max_iter, evaporation_rate=evaporation_rate)

# This statement is necessary in case data is not flushed yet.
del l
