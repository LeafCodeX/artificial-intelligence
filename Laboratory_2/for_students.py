from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import *


# Function to create initial population
def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]
    # List of lists, each inner list represents an individual in the population
    # Each individual is a list of boolean values representing the genes


# Function to calculate fitness of an individual
def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))


# Function to find the best individual in the population
def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    # The best individual is the one with the highest fitness
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness
    # Returns the individual with the highest fitness and its fitness value


items, knapsack_max_capacity = get_big()
print(items)
print(knapsack_max_capacity)


# Roulette wheel selection - two individuals selected based on their relative fitness
def roulette_wheel_selection(population, fitness):
    total_fitness = sum(fitness)
    relative_fitness = [f/total_fitness for f in fitness]
    return random.choices(population, weights=relative_fitness, k=2)


# (2.2.2 Crossover - Combine the genes of two parents to create new individuals)
def crossover(parent1, parent2):
    crossover_point = len(parent1) // 2
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


# (2.2.3 Mutation - Flip one gene with a certain probability)
def mutate(child, mutation_rate):
    if random.random() < mutation_rate:
        index = random.randint(0, len(child) - 1)
        child[index] = not child[index]
    return child


# (2.2.2 Crossover - Combine the genes of two parents to create new individuals)
def crossover_parents(parents):
    new_population = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i+1]
        child1, child2 = crossover(parent1, parent2)
        new_population.append(child1)
        new_population.append(child2)

    return new_population


# (2.2.3 Mutation - Select a random individual and flip one gene with a certain probability)
def mutate_population(population, mutation_rate):
    return [mutate(child, mutation_rate) for child in population]


population_size = 100
generations = 1000
n_elite = 20
n_selection = population_size - n_elite
mutation_rate = 1

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)    # 2.1.1 Create initial population
for _ in range(generations):
    population_history.append(population)

    new_population = []
    # Find best individual in current population
    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)

    fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_with_fitness = list(zip(population, fitnesses))
    population_with_fitness.sort(key=lambda x: x[1])

    elite_individuals = population_with_fitness[-n_elite:]
    new_population.extend([individual for individual, _ in elite_individuals])

    # (2.1.2 Select parents with roulette wheel selection)
    #parent_candidates = population_with_fitness[-n_selection:]
    parents = roulette_wheel_selection([individual for individual, _ in population_with_fitness],
                                       [fitness for _, fitness in population_with_fitness])
    #parents = [individual for individual, _ in parent_candidates]
    # (2.1.3 Create new population by performing crossover on all parents)
    children = crossover_parents(parents)
    # (2.1.4 Mutate the new population with a certain probability (mutation_rate)
    children = mutate_population(children, mutation_rate)
    new_population.extend(children)

    population = new_population

    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness

    # Append the best fitness of the current generation to the history of best fitness values
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()