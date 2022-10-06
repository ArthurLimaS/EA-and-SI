import numpy as np
import matplotlib.pyplot as plt

def random_index(indexes, pop_tam):
    r = indexes[0]
    while r in indexes:
        r = np.random.randint(0, pop_tam)
    
    return r

def run(graph, pop_ini, pop_tam, dim, max_gen, fitness_function, big_f=0.9, cr=0.8, lim_min=None, lim_max=None, rep=0):
    # Gráfico
    y = []

    pop = pop_ini.copy()

    fitness = np.zeros(shape=[pop_tam])
    fitness.astype(float)

    for i in range(pop_tam):
        fitness[i] = fitness_function(graph, pop[i])

    iteracao = 1
    while iteracao <= max_gen:
        old_pop = pop.copy()
        for i in range(pop_tam):
            r1 = random_index([i], pop_tam)
            r2 = random_index([i, r1], pop_tam)
            r3 = random_index([i, r1, r2], pop_tam)

            u = old_pop[r1] + (big_f * (old_pop[r2] - old_pop[r3]))

            s = np.random.rand(dim)
            u[s > cr] = old_pop[i][s > cr]
            u_fitness = fitness_function(graph, u)
            if u_fitness > fitness[i]:
                pop[i,:] = u
                fitness[i] = u_fitness

        ### verificar se os limites não foram estourados
        if lim_max is not None:
            pop[pop > lim_max] = lim_max[pop > lim_max]
        if lim_min is not None:
            pop[pop < lim_min] = lim_min[pop < lim_min]

        best_index = np.argmax(fitness)
        y.append(fitness_function(graph, pop[best_index]))
        print("GEN: {} / RES: {}".format(iteracao, fitness_function(graph, pop[best_index])))
        iteracao += 1

    best_index = np.argmax(fitness)

    return pop[best_index], y