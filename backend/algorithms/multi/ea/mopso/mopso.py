'''
Multi-Objective Particle Swarm Optimization (MOPSO)
Defined by Phil & GPT-4
'''

import numpy as np

# Define the Particle class
class Particle:
    def __init__(self, dimension):
        self.position = np.random.rand(dimension)
        self.velocity = np.random.rand(dimension) * 0.1
        self.best_position = np.copy(self.position)
        self.best_objectives = None

# Define the Repository class
class Repository:
    def __init__(self, max_size):
        self.solutions = []
        self.objectives = []
        self.max_size = max_size
    
    def add(self, solution, objectives):
        is_dominated = False
        to_remove = []
        for i, existing_objectives in enumerate(self.objectives):
            if self.dominates(objectives, existing_objectives):
                to_remove.append(i)
            elif self.dominates(existing_objectives, objectives):
                is_dominated = True
        for idx in reversed(to_remove):
            self.solutions.pop(idx)
            self.objectives.pop(idx)
        if not is_dominated:
            self.solutions.append(solution)
            self.objectives.append(objectives)
        if len(self.solutions) > self.max_size:
            self.prune()
    
    def dominates(self, obj1, obj2):
        # Check if obj1 dominates obj2
        return all(o1 <= o2 for o1, o2 in zip(obj1, obj2)) and any(o1 < o2 for o1, o2 in zip(obj1, obj2))
    
    def prune(self):
        # Simple pruning based on crowding distance (just a placeholder, more advanced methods can be used)
        distances = [0] * len(self.objectives)
        for i in range(len(self.objectives[0])):  # for each objective
            sorted_indices = np.argsort([obj[i] for obj in self.objectives])
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = float('inf')
            obj_range = self.objectives[sorted_indices[-1]][i] - self.objectives[sorted_indices[0]][i]
            for j in range(1, len(sorted_indices) - 1):
                distances[sorted_indices[j]] += (self.objectives[sorted_indices[j+1]][i] - self.objectives[sorted_indices[j-1]][i]) / obj_range
        while len(self.solutions) > self.max_size:
            idx_to_remove = np.argmin(distances)
            self.solutions.pop(idx_to_remove)
            self.objectives.pop(idx_to_remove)
            distances.pop(idx_to_remove)

# Define the MOPSO class
class MOPSO:
    def __init__(self, num_particles, dimension, max_iter, repo_size, w=0.5, c1=1.5, c2=1.5):
        self.num_particles = num_particles
        self.dimension = dimension
        self.max_iter = max_iter
        self.current_iter = 0
        self.particles = [Particle(dimension) for _ in range(num_particles)]
        self.repository = Repository(repo_size)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.asked_particles = []
    
    def ask(self):
        if self.current_iter < self.max_iter:
            particle = self.particles[self.current_iter % self.num_particles]
            self.asked_particles.append(particle)
            return particle.position
        else:
            return None  # No more particles to evaluate
    
    def tell(self, objectives):
        particle = self.asked_particles.pop(0)
        if particle.best_objectives is None or self.repository.dominates(objectives, particle.best_objectives):
            particle.best_position = np.copy(particle.position)
            particle.best_objectives = objectives
        self.repository.add(particle.position, objectives)
        if len(self.repository.solutions) > 0:
            leader = self.repository.solutions[np.random.randint(len(self.repository.solutions))]
        else:
            leader = particle.best_position
        particle.velocity = (self.w * particle.velocity +
                             self.c1 * np.random.rand(self.dimension) * (particle.best_position - particle.position) +
                             self.c2 * np.random.rand(self.dimension) * (leader - particle.position))
        particle.position += particle.velocity
        particle.position = np.clip(particle.position, 0, 1)  # Assuming all variables are in [0,1], can be modified
        self.current_iter += 1
    
    def get_pareto_front(self):
        return self.repository.solutions, self.repository.objectives

# Example usage:
optimizer = MOPSO(num_particles=10, dimension=4, max_iter=100, repo_size=20)
solutions, objectives = [], []

for _ in range(optimizer.max_iter):
    params = optimizer.ask()
    if params is not None:
        # Here, we would evaluate the objectives of the params.
        # For demonstration purposes, we'll just use two conflicting objectives.
        o1 = (params[0] ** 3 + params[1] ** 2) + params[2]**3 + params[3]**2
        print(o1)
        o2 = - (2**(1 + params[0]**2+params[1]**2))
        print(o2)
        objectives_mock = [o1, o2]
        optimizer.tell(objectives_mock)
        solutions.append(params)
        objectives.append(objectives_mock)

pareto_solutions, pareto_objectives = optimizer.get_pareto_front()
pareto_solutions, pareto_objectives
