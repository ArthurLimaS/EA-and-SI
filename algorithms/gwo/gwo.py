import numpy as np
from algorithms.gwo import gwo_auxiliary_funcions as gaf

def gwo(pop, pop_tam, dim, max_gen, lim_min, lim_max, fitness_funtion):
    # Inicializar variáveis a, A e C
    a = gaf.func_a(0, max_gen)
    big_a_alpha = gaf.func_big_a(a, pop_tam)
    big_a_beta = gaf.func_big_a(a, pop_tam)
    big_a_delta = gaf.func_big_a(a, pop_tam)
    big_c_alpha = gaf.func_big_c(pop_tam)
    big_c_beta = gaf.func_big_c(pop_tam)
    big_c_delta = gaf.func_big_c(pop_tam)

    # Calculo de fitness da população inicial
    fitness = np.zeros(shape=(pop_tam))

    for i in range(pop_tam):
        fitness[i] = fitness_funtion(pop[i])

    #print(fitness)

    # Determinar lobos alpha, beta e delta
    i_alpha = np.argmin(fitness)
    alpha = pop[i_alpha]

    beta_fitness = np.copy(fitness)
    beta_fitness[i_alpha] = 2 * 10**9
    i_beta = np.argmin(beta_fitness)
    beta = pop[i_beta]

    delta_fitness = np.copy(beta_fitness)
    delta_fitness[i_beta] = 2 * 10**9
    i_delta = np.argmin(delta_fitness)
    delta = pop[i_delta]

    # Plotar posição da primeira população
    #plot_figma(pop[:,0], pop[:,1], alpha, beta, delta)

    gen = 1
    while gen < max_gen:
        X1 = gaf.get_x(pop, pop_tam, dim, alpha, big_a_alpha, big_c_alpha)
        X2 = gaf.get_x(pop, pop_tam, dim, beta, big_a_beta, big_c_beta)
        X3 = gaf.get_x(pop, pop_tam, dim, delta, big_a_delta, big_c_delta)

        pop = (X1 + X2 + X3)/3

        ### verificar se os limites não foram estourados
        pop[pop > lim_max] = lim_max[pop > lim_max]
        pop[pop < lim_min] = lim_min[pop < lim_min]

        # atualizar variáveis a, A e C
        a = gaf.func_a(gen, max_gen)
        big_a_alpha = gaf.func_big_a(a, pop_tam)
        big_a_beta = gaf.func_big_a(a, pop_tam)
        big_a_delta = gaf.func_big_a(a, pop_tam)
        big_c_alpha = gaf.func_big_c(pop_tam)
        big_c_beta = gaf.func_big_c(pop_tam)
        big_c_delta = gaf.func_big_c(pop_tam)

        # Calculo de fitness
        for i in range(pop_tam):
            fitness[i] = fitness_funtion(pop[i])

        # Determinar lobos alpha, beta e delta
        i_alpha = np.argmin(fitness)
        alpha = pop[i_alpha]

        beta_fitness = np.copy(fitness)
        beta_fitness[i_alpha] = 2 * 10**9
        i_beta = np.argmin(beta_fitness)
        beta = pop[i_beta]

        delta_fitness = np.copy(beta_fitness)
        delta_fitness[i_beta] = 2 * 10**9
        i_delta = np.argmin(delta_fitness)
        delta = pop[i_delta]

        #if gen % 10 == 0:
            #plot_figma_iter(pop[:,0], pop[:,1], alpha, beta, delta)

        #print("GEN: {} / RES: {}".format(gen, sphere(alpha)))
        gen += 1