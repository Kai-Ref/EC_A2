import numpy as np
class ACO:
    def __init__(self, number_bits =50, evaporation_rate=0.1):
        self.number_bits = number_bits
        self.pheromone = np.ones((number_bits,2)) * 0.5  # pheromone for each bit being 0 or 1
        self.evaporation_rate = evaporation_rate
        self.previous_best = None

    def __call__(self):
        probs = self.pheromone[:, 0] / (self.pheromone.sum(axis=1) + 1e-10)
        solution = (np.random.rand(self.number_bits) > probs).astype(int)
        return solution

    def update_pheromone(self, best_solution):
        evaporation = (1-self.evaporation_rate)  * self.pheromone
        self.pheromone = evaporation
        self.pheromone[np.arange(self.number_bits), best_solution] += self.evaporation_rate
        self.pheromone = np.clip(self.pheromone, 1/self.number_bits, 1 - 1/self.number_bits)
