import numpy as np


class ACO:
    def __init__(self, n_variables=50, evaporation_rate=0.1, num_ants=10, budget=100000, pheromone_min=1e-6,
                 pheromone_max=100, epsilon=0.01):
        self.n_variables = n_variables            # Dimension of the problem
        self.evaporation_rate = evaporation_rate  # Evaporation rate of pheromone
        self.num_ants = num_ants                  # Number of ants
        self.budget = budget                      # Budget (number of function evaluations)

        # Pheromone bounds
        self.pheromone_min = pheromone_min
        self.pheromone_max = pheromone_max

        self.epsilon = epsilon

        self.pheromone = np.ones((self.n_variables, 2))

        # Create the ants
        self.ants = []
        for _ in range(self.num_ants):
            self.ants.append(Ant(n_variables=self.n_variables, epsilon=self.epsilon))

    def update_pheromone(self, x_opt, f_opt, optimum):
        if x_opt is not None:
            # Create pheromone update matrix
            updates = np.zeros_like(self.pheromone)
            updates[np.arange(self.n_variables), x_opt.astype(int)] += (f_opt / optimum)

            # Update and clip to range pheromone values
            self.pheromone += updates
            self.pheromone = np.clip((1 - self.evaporation_rate) * self.pheromone, 1e-6, 100)

    def run(self, func, x_opt, f_opt, optimum):
        # Main loop of ACO
        for _ in range(self.budget):
            # For each ant, create a solution and evaluate it
            for ant in self.ants:
                x = ant.create_solution(pheromone=self.pheromone)

                f = func(x)

                # If it is the best solution so far, update the best solution
                if f > f_opt:
                    f_opt = f
                    x_opt = x

            # If we find the optimum, stop the algorithm early
            if f_opt >= optimum:
                break

            # Update pheromone
            self.update_pheromone(x_opt, f_opt, optimum)


class Ant:
    def __init__(self, n_variables, epsilon=0.01):
        self.n_variables = n_variables      # Dimension of the problem
        self.epsilon = epsilon              # Exploration factor

    def create_solution(self, pheromone):
        probs = pheromone[:, 1] / pheromone.sum(axis=1)
        probs = (1 - self.epsilon) * probs + self.epsilon * 0.5

        solution = (np.random.rand(self.n_variables) < probs).astype(int)
        return solution
