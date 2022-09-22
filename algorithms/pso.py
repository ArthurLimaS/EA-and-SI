import numpy as np

def run(graph, pop, pop_tam, dim, max_gen, fitness_funtion, lim_min, lim_max, c_1=2, c_2=2, w=0.7):
    velocities = (np.random.rand(pop_tam, dim) * ((lim_max - lim_min) / 10)) + lim_min

    # Fitness evaluation
    fitness = np.zeros(shape=[pop_tam])

    for i in range(pop_tam):
        fitness[i] = fitness_funtion(graph, pop[i])

    best_index = np.argmax(fitness)

    local_best_population = pop.copy()
    local_best_fitness = fitness.copy()

    for generation in range(max_gen):
        # Update all velocities
        cr_1 = np.random.rand(pop_tam, dim) * c_1
        cr_2 = np.random.rand(pop_tam, dim) * c_2

        tmp_best_individual = np.zeros(shape=[pop_tam, dim])
        tmp_best_individual[0:pop_tam, :] = local_best_population[best_index, :]

        velocities = (w * velocities) \
                     + (cr_1 * (local_best_population - pop)) \
                     + (cr_2 * (tmp_best_individual - pop))

        np.add(pop, velocities, out=pop, casting="unsafe")

        # Out-bounded individual treatment
        pop[pop > lim_max] = lim_max[pop > lim_max]
        pop[pop < lim_min] = lim_min[pop < lim_min]

        # Fitness Evaluation
        for i in range(pop_tam):
            fitness[i] = fitness_funtion(graph, pop[i])

        # Update the Local Best Population
        tmp_best_indices = fitness > local_best_fitness
        local_best_fitness[tmp_best_indices] = fitness[tmp_best_indices]
        local_best_population[tmp_best_indices, :] = pop[tmp_best_indices, :]

        best_index = np.argmax(local_best_fitness)

        print('Generation ' + str(generation) + ": " + str(local_best_fitness[best_index]))
    return pop[best_index]