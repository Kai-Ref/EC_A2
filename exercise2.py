from ioh import get_problem, ProblemClass
from ioh import logger
import sys
import argparse
import numpy as np

def RS_EA(func, budget = None):
    if budget is None:
        budget = int(100000)

    optimum = func.optimum.y
    print(f"function optimum: {optimum}")
    for r in range(10):
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


def RLS_EA(func, budget = None):
    if budget is None:
        budget = int(100000)

    optimum = func.optimum.y
    print(f"function optimum: {optimum}")
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

parser = argparse.ArgumentParser(prog="PBO Exercise 2 Script", description="Tests RLS, (1+1), and Random Search on various fitness functions")
parser.add_argument('function')


if __name__ == "__main__":
    args = parser.parse_args()
    if(args.function == "one"):
        log = logger.Analyzer(root="data/ex2", 
        folder_name="run", 
        algorithm_name="(1+1)-EA", 
        algorithm_info="tests in python with the (1+1) EA Algorithm")

        om.attach_logger(log)
        one_one_EA(om)

        lo.attach_logger(log)
        one_one_EA(lo)

        lhw.attach_logger(log)
        one_one_EA(lhw)

        labs.attach_logger(log)
        one_one_EA(labs)

        nqp.attach_logger(log)
        one_one_EA(nqp)

        ct.attach_logger(log)
        one_one_EA(ct)

        nkl.attach_logger(log)
        one_one_EA(nkl)

        del log
        
    elif(args.function == "rls"):
        log = logger.Analyzer(root="data/ex2", 
        folder_name="run", 
        algorithm_name="RLS-EA", 
        algorithm_info="tests in python with the RLS EA Algorithm")

        om.attach_logger(log)
        RLS_EA(om)

        lo.attach_logger(log)
        RLS_EA(lo)

        lhw.attach_logger(log)
        RLS_EA(lhw)

        labs.attach_logger(log)
        RLS_EA(labs)

        nqp.attach_logger(log)
        RLS_EA(nqp)

        ct.attach_logger(log)
        RLS_EA(ct)

        nkl.attach_logger(log)
        RLS_EA(nkl)

        del log


    elif(args.function == "ran"):
        log = logger.Analyzer(root="data/ex2", 
        folder_name="run", 
        algorithm_name="Random Search", 
        algorithm_info="tests in python with the Random Search Algorithm")

        om.attach_logger(log)
        RS_EA(om)

        lo.attach_logger(log)
        RS_EA(lo)

        lhw.attach_logger(log)
        RS_EA(lhw)

        labs.attach_logger(log)
        RS_EA(labs)

        nqp.attach_logger(log)
        RS_EA(nqp)

        ct.attach_logger(log)
        RS_EA(ct)

        nkl.attach_logger(log)
        RS_EA(nkl)

        del log
    else:
        print("Please input a correct function - one, rls, or ran")
        


    
