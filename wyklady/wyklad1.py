import pygad
import numpy

# nasz problem
S = [1, 2, 3, 6, 10, 17, 25, 29, 30, 41, 51, 60, 70, 79, 90]

# definiujemy parametry chromosomu
# geny to liczby: 0 lub 1
gene_space = [0, 1]
# gene_space = [0,1,2,3]
# gene_space = {'low': 0, 'high': 1, 'step': 0.1}

# długość chromosomu
num_genes = len(S)

# ile chromosomów w populacji
# solution_per_population = 10
sol_per_pop = 10

# ile pokoleń ( generacji )
num_generations = 30

# selekcja
parent_selection_type = "sss"
num_parents_mating = 5
keep_parents = 2

# krzyżowanie
crossover_type = "single_point"
# crossover_type = "double_points"
# crossover_type = "uniform"

# mutacja
mutation_type = "random"
# mutation_type = "swap"
# mutation_type = "scramble"
# mutation_type = "inversion"
mutation_percent_genes = 8


# funkcja celu
def fitness_func(instance, solution, solution_idx):
    sum1 = numpy.sum(solution * S)
    solution_invert = 1 - solution
    sum2 = numpy.sum(solution_invert * S)
    fitness = -abs(sum1 - sum2)
    return fitness


fitness_function = fitness_func

ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

ga_instance.plot_fitness()