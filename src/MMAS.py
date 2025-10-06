from ioh import get_problem, ProblemClass
from ioh import logger
import argparse
import time
from tqdm import tqdm
from aco import ACO


# MMAS implementation
def mmas(func, budget=None, evaporation_rate=0.02, strict=False):
    if budget is None:
        budget = int(func.meta_data.n_variables * func.meta_data.n_variables * 50)

    if func.meta_data.problem_id == 18 and func.meta_data.n_variables == 32:
        optimum = 8
    else:
        optimum = func.optimum.y

    start = time.time()
    average_time_70 = 0
    for _ in tqdm(range(10), desc=f"Runs on F{func.meta_data.problem_id}"):
        aco = ACO(number_bits=func.meta_data.n_variables, evaporation_rate=evaporation_rate)
        f_opt = 0.0
        time_70 = None
        start_this_run = time.time()
        x_opt = None
        for _ in range(budget):
            x = aco()
            f = func(x)

            # --- difference between MMAS and MMAS* ---
            if (not strict and (f >= f_opt or x_opt is None)) or (strict and (f > f_opt or x_opt is None)):
                f_opt = f
                x_opt = x
            if f_opt >= optimum:
                break
            if f_opt >= 0.7 * optimum and time_70 is None:
                time_70 = time.time() - start_this_run
                average_time_70 += time_70
            aco.update_pheromone(x_opt)
        func.reset()
    end = time.time()
    print(f"[F{func.meta_data.problem_id}] Total time for 100 runs: {end - start:.2f}s,"
          f" avg per run: {(end - start) / 100:.2f}s,"
          f" evaporation: {evaporation_rate},"
          f" avg time to reach 70%: {average_time_70 / 100:.2f}s")
    return f_opt, x_opt


# Parameters
n = 100
max_iter = 100000

parser = argparse.ArgumentParser()
parser.add_argument('--evaporation_rate', type=float, default=1, help='Evaporation rate for MMAS')
args = parser.parse_args()
evaporation_rate = args.evaporation_rate

# Logger
l = logger.Analyzer(
    root="data",
    folder_name=f"mmas_{evaporation_rate}_evap",
    algorithm_name=f"mmas_{evaporation_rate}",
    algorithm_info=f"test of IOHexperimenter in python, evaporation_rate={evaporation_rate}",
)

# List of problem IDs
problem_ids = [1, 2, 3, 18, 23, 24, 25]

# Run MMAS on each problem
for fid in problem_ids:
    func = get_problem(fid=fid, dimension=n, instance=1, problem_class=ProblemClass.PBO)
    func.attach_logger(l)
    mmas(func, budget=max_iter, evaporation_rate=evaporation_rate, strict=False)

# Flush logger
del l

# Logger
l = logger.Analyzer(
    root="data",
    folder_name=f"mmas_star_{evaporation_rate}_evap",
    algorithm_name=f"mmas_star_{evaporation_rate}",
    algorithm_info=f"test of IOHexperimenter in python, evaporation_rate={evaporation_rate}",
)

# List of problem IDs
problem_ids = [1, 2, 3, 18, 23, 24, 25]

# Run MMAS on each problem
for fid in problem_ids:
    func = get_problem(fid=fid, dimension=n, instance=1, problem_class=ProblemClass.PBO)
    func.attach_logger(l)
    mmas(func, budget=max_iter, evaporation_rate=evaporation_rate, strict=True)

# Flush logger
del l