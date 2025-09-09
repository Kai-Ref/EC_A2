import numpy as np
class ACO:
    def __init__(self, number_bits =50, evaporation_rate=0.1):
        self.number_bits = number_bits
        self.pheromone = np.ones((number_bits,2)) * 0.5  # pheromone for each bit being 0 or 1
        self.evaporation_rate = evaporation_rate

    def __call__(self):
        probs = self.pheromone[:, 0] / (self.pheromone.sum(axis=1) + 1e-10)
        solution = (np.random.rand(self.number_bits) < probs).astype(int)
        return solution

    def update_pheromone(self, best_solution, reward):
        self.pheromone =(1-self.evaporation_rate)  * self.pheromone
        self.pheromone[np.arange(self.number_bits), best_solution] += reward * self.evaporation_rate