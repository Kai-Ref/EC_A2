from ioh import get_problem, ProblemClass
from ioh import logger
import sys
import numpy as np

# Please replace this `random search` by your `genetic algorithm`.
def random_search(func, budget = None):
    # budget of each run: 50n^2
    if budget is None:
        budget = int(100000)

    optimum = func.optimum.y
    print(optimum)
    # 10 independent runs for each algorithm on each problem.
    for r in range(1000):
        f_opt = sys.float_info.min
        x_opt = None
        for i in range(budget):
            x = np.random.randint(2, size = func.meta_data.n_variables)
            f = func(x)
            if f > f_opt:
                f_opt = f
                x_opt = x
            if f_opt >= optimum:
                break
        func.reset()
    return f_opt, x_opt


#I just copied the random search and changed the function
def RLS_EA(func, budget = None):
    # Im pretty sure budget is number of iterations
    if budget is None:
        budget = int(100000)

    optimum = func.optimum.y
    print(optimum)
    # 10 independent runs for each algorithm on each problem.
    for r in range(10):
        x_opt = np.random.randint(2, size = func.meta_data.n_variables)
        x = x_opt
        f_opt = func(x_opt)
        for i in range(budget):
            #choose a random bit to flip
            flip = np.random.randint(func.meta_data.n_variables)
            #flip that bits
            if x[flip] == 1:
                x[flip] = 0
            else:
                x[flip] = 1
            
            f = func(x)
            if f > f_opt:
                f_opt = f
                x_opt = x
            if f_opt >= optimum:
                break

            
        func.reset()
    return f_opt, x_opt  

def one_one_EA(func, budget = None):
    # Im pretty sure budget is number of iterations
    if budget is None:
        budget = int(100000)

    optimum = func.optimum.y
    print(optimum)
    # 10 independent runs for each algorithm on each problem.
    for r in range(10):
        x_opt = np.random.randint(2, size = func.meta_data.n_variables)
        x = x_opt
        f_opt = func(x_opt)
        for i in range(budget):
            #create a random array and see if any of their values have probability 1/n
            flips = np.random.rand(func.meta_data.n_variables) <= (1/func.meta_data.n_variables)
            #flip those bits
            x = np.bitwise_xor(x, flips)
            f = func(x)
            if f > f_opt:
                f_opt = f
                x_opt = x
            if f_opt >= optimum:
                break

            
        func.reset()
    return f_opt, x_opt  

# Declaration of problems to be tested.
om = get_problem(fid = 1, dimension=100, instance=1, problem_class = ProblemClass.PBO)
lo = get_problem(fid = 2, dimension=100, instance=1, problem_class = ProblemClass.PBO)
lhw = get_problem(fid = 3, dimension=100, instance=1, problem_class = ProblemClass.PBO)
labs = get_problem(fid = 18, dimension=100, instance=1, problem_class = ProblemClass.PBO)
nqp = get_problem(fid = 23, dimension=100, instance=1, problem_class = ProblemClass.PBO)
ct = get_problem(fid = 24, dimension=100, instance=1, problem_class = ProblemClass.PBO)
nkl = get_problem(fid = 25, dimension=100, instance=1, problem_class = ProblemClass.PBO)

# Create default logger compatible with IOHanalyzer
# `root` indicates where the output files are stored.
# `folder_name` is the name of the folder containing all output. You should compress this folder and upload it to IOHanalyzer
l_rls = logger.Analyzer(root="data", 
    folder_name="run", 
    algorithm_name="RLS-EA", 
    algorithm_info="test of IOHexperimenter in python with the RLS EA")


om.attach_logger(l_rls)
RLS_EA(om)

lo.attach_logger(l_rls)
RLS_EA(lo)

lhw.attach_logger(l_rls)
RLS_EA(lhw)

labs.attach_logger(l_rls)
RLS_EA(labs)

nqp.attach_logger(l_rls)
RLS_EA(nqp)

ct.attach_logger(l_rls)
RLS_EA(ct)

nkl.attach_logger(l_rls)
RLS_EA(nkl)

# This statemenet is necessary in case data is not flushed yet.
del l_rls
