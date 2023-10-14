import math
import pygad
import numpy

S = [
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
]


def fitness_function(i, solution, sid):
    def move_pos(pos, move):
        if move == 0: return [pos[0], pos[1] - 1]
        if move == 1: return [pos[0] + 1, pos[1]]
        if move == 2: return [pos[0], pos[1] + 1]
        if move == 3: return [pos[0] - 1, pos[1]]

    def is_corrent_move(pos):
        if S[pos[1]][pos[0]] == 0:
            return True
        else:
            return False

    pos = [0, 0]
    for i in range(len(solution)):
        prev_post = [
            pos[0],
            pos[1]
        ]
        pos = move_pos(pos, solution[i])
        if pos[0] == 9 and pos[1] == 9:
            break
        if pos[0] > 9 or pos[1] > 9 or pos[0] < 0 or pos[1] < 0:
            return -1000
        if not is_corrent_move(pos):
            pos = prev_post

    return pos[0] + pos[1]


ga_instance = pygad.GA(
    gene_space=[0, 1, 2, 3],
    num_generations=1000,
    num_parents_mating=20,
    fitness_func=fitness_function,
    sol_per_pop=100,
    num_genes=30,
    parent_selection_type="sss",
    keep_parents=2,
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=8,

    stop_criteria=[
        "reach_18"
    ]
)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Rozwiązanie znaleziono po:")
print(ga_instance.generations_completed)

ga_instance.plot_fitness()
