#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Dict
import random
import math

class GA:
    """Genetic Algorithm as per paper specifications"""
    
    def __init__(self, 
                 population_size: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 max_generations: int = 100,
                 bounds: List[Tuple[float, float]] = None):
        
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.bounds = bounds
        self.dimension = len(bounds) if bounds else 2
        
        # Results tracking
        self.best_history = []
        self.best_individual = None
        self.best_fitness = float('inf')
        self.generation = 0
    
    def initialize_population(self):
        """Initialize random population"""
        population = []
        for _ in range(self.population_size):
            individual = []
            for i in range(self.dimension):
                lower, upper = self.bounds[i]
                individual.append(random.uniform(lower, upper))
            population.append(np.array(individual))
        return np.array(population)
    
    def evaluate_population(self, population, objective_function):
        """Evaluate fitness of all individuals"""
        fitness = []
        for individual in population:
            fitness.append(objective_function(individual))
        return np.array(fitness)
    
    def selection(self, population, fitness):
        """Tournament selection"""
        selected = []
        for _ in range(self.population_size):
            # Tournament size of 3
            tournament_indices = random.sample(range(len(population)), 3)
            tournament_fitness = fitness[tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        return np.array(selected)
    
    def crossover(self, parent1, parent2):
        """Uniform crossover"""
        if random.random() < self.crossover_rate:
            child1 = parent1.copy()
            child2 = parent2.copy()
            
            for i in range(self.dimension):
                if random.random() < 0.5:
                    child1[i], child2[i] = child2[i], child1[i]
            
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    
    def mutation(self, individual):
        """Gaussian mutation"""
        mutated = individual.copy()
        for i in range(self.dimension):
            if random.random() < self.mutation_rate:
                lower, upper = self.bounds[i]
                std = (upper - lower) * 0.1  # 10% of range as std
                noise = np.random.normal(0, std)
                mutated[i] = np.clip(mutated[i] + noise, lower, upper)
        return mutated
    
    def optimize(self, objective_function, verbose=True):
        """Main GA optimization loop"""
        if verbose:
            print("=" * 60)
            print("🧬 GENETIC ALGORITHM (Paper Implementation)")
            print("=" * 60)
            print(f"Population size: {self.population_size}")
            print(f"Crossover rate: {self.crossover_rate}")
            print(f"Mutation rate: {self.mutation_rate}")
            print(f"Max generations: {self.max_generations}")
            print("-" * 60)
        
        # Initialize population
        population = self.initialize_population()
        
        for generation in range(self.max_generations):
            self.generation = generation
            
            # Evaluate population
            fitness = self.evaluate_population(population, objective_function)
            
            # Update best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_fitness:
                self.best_fitness = fitness[best_idx]
                self.best_individual = population[best_idx].copy()
            
            self.best_history.append(self.best_fitness)
            
            if verbose and generation % 20 == 0:
                print(f"Generation {generation}: Best = {self.best_fitness:.6f}")
            
            # Selection
            selected = self.selection(population, fitness)
            
            # Create new population
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = selected[i]
                parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                new_population.extend([child1, child2])
            
            population = np.array(new_population[:self.population_size])
        
        if verbose:
            print(f"\n✅ GA COMPLETED!")
            print(f"Best fitness: {self.best_fitness:.6f}")
            print(f"Best individual: {self.best_individual}")
        
        return {
            'best_x': self.best_individual,
            'best_y': self.best_fitness,
            'history': self.best_history,
            'generations': self.generation + 1
        }

class PSO:
    """Particle Swarm Optimization as per paper specifications"""
    
    def __init__(self, 
                 n_particles: int = 50,
                 w: float = 0.7,  # inertia weight
                 c1: float = 1.5,  # cognitive parameter
                 c2: float = 2.0,  # social parameter
                 max_iterations: int = 100,
                 bounds: List[Tuple[float, float]] = None):
        
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.bounds = bounds
        self.dimension = len(bounds) if bounds else 2
        
        # Results tracking
        self.best_history = []
        self.gbest_position = None
        self.gbest_value = float('inf')
        self.iteration = 0
    
    def initialize_particles(self):
        """Initialize particles with random positions and velocities"""
        positions = []
        velocities = []
        
        for _ in range(self.n_particles):
            position = []
            velocity = []
            
            for i in range(self.dimension):
                lower, upper = self.bounds[i]
                position.append(random.uniform(lower, upper))
                # Initialize velocity as 10% of search space
                velocity.append(random.uniform(-(upper-lower)*0.1, (upper-lower)*0.1))
            
            positions.append(position)
            velocities.append(velocity)
        
        return np.array(positions), np.array(velocities)
    
    def optimize(self, objective_function, verbose=True):
        """Main PSO optimization loop"""
        if verbose:
            print("=" * 60)
            print("🐝 PARTICLE SWARM OPTIMIZATION (Paper Implementation)")
            print("=" * 60)
            print(f"Particles: {self.n_particles}")
            print(f"Inertia weight (w): {self.w}")
            print(f"Cognitive parameter (c1): {self.c1}")
            print(f"Social parameter (c2): {self.c2}")
            print(f"Max iterations: {self.max_iterations}")
            print("-" * 60)
        
        # Initialize particles
        positions, velocities = self.initialize_particles()
        pbest_positions = positions.copy()
        pbest_values = np.array([objective_function(pos) for pos in positions])
        
        # Initialize global best
        best_idx = np.argmin(pbest_values)
        self.gbest_position = pbest_positions[best_idx].copy()
        self.gbest_value = pbest_values[best_idx]
        
        for iteration in range(self.max_iterations):
            self.iteration = iteration
            
            for i in range(self.n_particles):
                # Evaluate current position
                current_value = objective_function(positions[i])
                
                # Update personal best
                if current_value < pbest_values[i]:
                    pbest_values[i] = current_value
                    pbest_positions[i] = positions[i].copy()
                
                # Update global best
                if current_value < self.gbest_value:
                    self.gbest_value = current_value
                    self.gbest_position = positions[i].copy()
            
            self.best_history.append(self.gbest_value)
            
            if verbose and iteration % 20 == 0:
                print(f"Iteration {iteration}: Best = {self.gbest_value:.6f}")
            
            # Update velocities and positions
            for i in range(self.n_particles):
                r1, r2 = random.random(), random.random()
                
                # Update velocity
                cognitive = self.c1 * r1 * (pbest_positions[i] - positions[i])
                social = self.c2 * r2 * (self.gbest_position - positions[i])
                velocities[i] = self.w * velocities[i] + cognitive + social
                
                # Update position
                positions[i] += velocities[i]
                
                # Apply bounds
                for j in range(self.dimension):
                    lower, upper = self.bounds[j]
                    positions[i][j] = np.clip(positions[i][j], lower, upper)
        
        if verbose:
            print(f"\n✅ PSO COMPLETED!")
            print(f"Best value: {self.gbest_value:.6f}")
            print(f"Best position: {self.gbest_position}")
        
        return {
            'best_x': self.gbest_position,
            'best_y': self.gbest_value,
            'history': self.best_history,
            'iterations': self.iteration + 1
        }

class ACO:
    """Ant Colony Optimization as per paper specifications"""
    
    def __init__(self, 
                 n_ants: int = 30,
                 evaporation_rate: float = 0.1,
                 alpha: float = 1.0,  # pheromone importance
                 beta: float = 2.0,   # heuristic importance
                 max_iterations: int = 100,
                 bounds: List[Tuple[float, float]] = None):
        
        self.n_ants = n_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.max_iterations = max_iterations
        self.bounds = bounds
        self.dimension = len(bounds) if bounds else 2
        
        # Discretize search space
        self.n_intervals = 50  # Number of intervals per dimension
        self.pheromone = np.ones((self.dimension, self.n_intervals))
        
        # Results tracking
        self.best_history = []
        self.best_solution = None
        self.best_value = float('inf')
        self.iteration = 0
    
    def continuous_to_discrete(self, continuous_solution):
        """Convert continuous solution to discrete indices"""
        discrete_solution = []
        for i in range(self.dimension):
            lower, upper = self.bounds[i]
            value = continuous_solution[i]
            # Normalize to [0, 1]
            normalized = (value - lower) / (upper - lower)
            # Convert to interval index
            interval_idx = int(normalized * (self.n_intervals - 1))
            interval_idx = np.clip(interval_idx, 0, self.n_intervals - 1)
            discrete_solution.append(interval_idx)
        return discrete_solution
    
    def discrete_to_continuous(self, discrete_solution):
        """Convert discrete indices to continuous solution"""
        continuous_solution = []
        for i in range(self.dimension):
            lower, upper = self.bounds[i]
            interval_idx = discrete_solution[i]
            # Convert interval index to continuous value
            normalized = interval_idx / (self.n_intervals - 1)
            value = lower + normalized * (upper - lower)
            continuous_solution.append(value)
        return continuous_solution
    
    def construct_solution(self, ant_idx):
        """Construct solution for one ant"""
        solution = []
        
        for dim in range(self.dimension):
            # Calculate probabilities for this dimension
            probabilities = (self.pheromone[dim] ** self.alpha) * \
                          (np.ones(self.n_intervals) ** self.beta)
            probabilities = probabilities / np.sum(probabilities)
            
            # Select interval based on probabilities
            cumulative_probs = np.cumsum(probabilities)
            rand_val = random.random()
            selected_interval = np.searchsorted(cumulative_probs, rand_val)
            selected_interval = min(selected_interval, self.n_intervals - 1)
            
            solution.append(selected_interval)
        
        return solution
    
    def update_pheromone(self, solutions, values):
        """Update pheromone trails"""
        # Evaporation
        self.pheromone *= (1 - self.evaporation_rate)
        
        # Add pheromone based on solution quality
        for solution, value in zip(solutions, values):
            pheromone_deposit = 1.0 / (1.0 + value)  # Inverse of objective value
            for dim in range(self.dimension):
                interval = solution[dim]
                self.pheromone[dim, interval] += pheromone_deposit
    
    def optimize(self, objective_function, verbose=True):
        """Main ACO optimization loop"""
        if verbose:
            print("=" * 60)
            print("🐜 ANT COLONY OPTIMIZATION (Paper Implementation)")
            print("=" * 60)
            print(f"Ants: {self.n_ants}")
            print(f"Evaporation rate: {self.evaporation_rate}")
            print(f"Alpha (pheromone): {self.alpha}")
            print(f"Beta (heuristic): {self.beta}")
            print(f"Max iterations: {self.max_iterations}")
            print("-" * 60)
        
        for iteration in range(self.max_iterations):
            self.iteration = iteration
            
            # Construct solutions for all ants
            solutions = []
            values = []
            
            for ant in range(self.n_ants):
                discrete_solution = self.construct_solution(ant)
                continuous_solution = self.discrete_to_continuous(discrete_solution)
                value = objective_function(continuous_solution)
                
                solutions.append(discrete_solution)
                values.append(value)
                
                # Update best solution
                if value < self.best_value:
                    self.best_value = value
                    self.best_solution = continuous_solution.copy()
            
            # Update pheromone
            self.update_pheromone(solutions, values)
            
            self.best_history.append(self.best_value)
            
            if verbose and iteration % 20 == 0:
                print(f"Iteration {iteration}: Best = {self.best_value:.6f}")
        
        if verbose:
            print(f"\n✅ ACO COMPLETED!")
            print(f"Best value: {self.best_value:.6f}")
            print(f"Best solution: {self.best_solution}")
        
        return {
            'best_x': self.best_solution,
            'best_y': self.best_value,
            'history': self.best_history,
            'iterations': self.iteration + 1
        }

def test_algorithms():
    """Test all algorithms on a simple function"""
    
    def sphere(x):
        return np.sum(x**2)
    
    bounds = [(-5, 5), (-5, 5)]
    
    print("Testing algorithms on Sphere function...")
    
    # Test GA
    ga = GA(bounds=bounds, max_generations=50)
    ga_result = ga.optimize(sphere, verbose=False)
    
    # Test PSO
    pso = PSO(bounds=bounds, max_iterations=50)
    pso_result = pso.optimize(sphere, verbose=False)
    
    # Test ACO
    aco = ACO(bounds=bounds, max_iterations=50)
    aco_result = aco.optimize(sphere, verbose=False)
    
    print(f"GA best: {ga_result['best_y']:.6f}")
    print(f"PSO best: {pso_result['best_y']:.6f}")
    print(f"ACO best: {aco_result['best_y']:.6f}")

if __name__ == "__main__":
    test_algorithms()
