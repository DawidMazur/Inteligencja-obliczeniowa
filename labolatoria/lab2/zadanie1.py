import pygad
import numpy

S = [
    {
        "przedmiot": "zegar",
        "wartosc": 100,
        "waga": 7,
    },
    {
        "przedmiot": "obraz-pejzaz",
        "wartosc": 300,
        "waga": 7,
    },
    {
        "przedmiot": "obraz-portret",
        "wartosc": 200,
        "waga": 6,
    },
    {
        "przedmiot": "radio",
        "wartosc": 40,
        "waga": 2,
    },
    {
        "przedmiot": "laptop",
        "wartosc": 500,
        "waga": 5,
    },
    {
        "przedmiot": "lampka nocna",
        "wartosc": 70,
        "waga": 6,
    },
    {
        "przedmiot": "srebrne sztucce",
        "wartosc": 100,
        "waga": 1,
    },
    {
        "przedmiot": "porcelana",
        "wartosc": 250,
        "waga": 3,
    },
    {
        "przedmiot": "figura z brazu",
        "wartosc": 300,
        "waga": 10,
    },
    {
        "przedmiot": "skorzana torebka",
        "wartosc": 280,
        "waga": 3,
    },
    {
        "przedmiot": "odkurzacz",
        "wartosc": 300,
        "waga": 15,
    }
]
max_weight = 25

gene_space = [0, 1]

def fitness_func(instance, solution, solution_idx):
    waga = 0
    wartosc = 0
    for i in range(len(solution)):
        if solution[i] == 1:
            waga += S[i]["waga"]
            wartosc += S[i]["wartosc"]

    if(waga > max_weight) :
        return 0

    return wartosc




sol_per_pop = 10
num_genes = len(S)

num_parents_mating = 5
num_generations = 30
keep_parents = 2

parent_selection_type = "sss"

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 8


ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

#uruchomienie algorytmu
ga_instance.run()

#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

wybrane_przedmioty = ""
wartosc = 0

for i in range(len(solution)):
    if solution[i] == 1:
        wybrane_przedmioty += S[i]["przedmiot"] + ", "
        wartosc += S[i]["wartosc"]
print("Wybrane przedmioty: " + wybrane_przedmioty)

#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()

