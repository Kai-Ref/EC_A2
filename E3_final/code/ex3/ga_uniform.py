from ioh import get_problem, ProblemClass, logger
import numpy as np
import sys

rng = np.random.default_rng()

def tournament_select(pop, fitness, k=3):
    idxs = rng.integers(0, len(pop), size=k)
    best = idxs[0]
    for i in idxs[1:]:
        if fitness[i] > fitness[best]:
            best = i
    return best

def uniform_crossover(p1, p2):
    # 0/1 int arrays
    mask = rng.random(p1.size) < 0.5
    return np.where(mask, p1, p2).astype(np.int64)

def bitflip_mutation(x, p):
    flips = rng.random(x.size) < p
    return np.bitwise_xor(x, flips).astype(np.int64)

def ga_uniform(func,
               budget=100_000,
               mu=20,              # parents per brief
               lambd=20,           # offspring per gen
               pc=1.0,             # crossover prob
               pm=None,            # default 1/n
               tour_k=3,
               elitism=True,
               runs=10):
    n = func.meta_data.n_variables
    if pm is None:
        pm = 1.0 / n

    optimum = func.optimum.y
    for r in range(runs):
        # init parents
        P = rng.integers(0, 2, size=(mu, n), dtype=np.int64)
        F = np.empty(mu, dtype=float)
        evals = 0
        for i in range(mu):
            F[i] = func(P[i]); evals += 1
        best_f = F.max()
        if best_f >= optimum:
            func.reset(); continue

        # gens til budget
        while evals < budget:
            # produce offspring
            O = np.empty((lambd, n), dtype=np.int64)
            for j in range(lambd):
                p1 = P[tournament_select(P, F, k=tour_k)]
                p2 = P[tournament_select(P, F, k=tour_k)]
                child = uniform_crossover(p1, p2) if rng.random() < pc else p1.copy()
                child = bitflip_mutation(child, pm)
                O[j] = child

            # evaluate offspring
            FO = np.empty(lambd, dtype=float)
            for j in range(lambd):
                if evals >= budget:
                    break
                FO[j] = func(O[j]); evals += 1

            # elitism survivor selection 
            if elitism:
                P_all = np.vstack([P, O])
                F_all = np.concatenate([F, FO])
                idx = np.argsort(-F_all)[:mu]
                P, F = P_all[idx], F_all[idx]
            else:
                # preseve best parent + best offspring
                bp = np.argmax(F)
                top_off = np.argsort(-FO)[:mu-1]
                P = np.vstack([P[bp], O[top_off]])
                F = np.concatenate([[F[bp]], FO[top_off]])

            if F.max() > best_f:
                best_f = F.max()
            if best_f >= optimum:
                break

        func.reset()

if __name__ == "__main__":
    problems = [
        get_problem(fid=1,  dimension=100, instance=1, problem_class=ProblemClass.PBO),
        get_problem(fid=2,  dimension=100, instance=1, problem_class=ProblemClass.PBO),
        get_problem(fid=3,  dimension=100, instance=1, problem_class=ProblemClass.PBO),
        get_problem(fid=18, dimension=100, instance=1, problem_class=ProblemClass.PBO),
        get_problem(fid=23, dimension=100, instance=1, problem_class=ProblemClass.PBO),
        get_problem(fid=24, dimension=100, instance=1, problem_class=ProblemClass.PBO),
        get_problem(fid=25, dimension=100, instance=1, problem_class=ProblemClass.PBO),
    ]

    l = logger.Analyzer(
        root="data",
        folder_name="run-ga",
        algorithm_name="GA_uniform",
        algorithm_info="Uniform crossover; pm=1/n; mu=lambda=20; tournament k=3; (mu+lambda) elitist"
    )

    for f in problems:
        f.attach_logger(l)
        ga_uniform(
            f,
            budget=100_000, 
            mu=20, lambd=20, pc=1.0, pm=None, tour_k=3, elitism=True, runs=10
        )

    del l
