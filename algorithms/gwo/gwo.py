import numpy as np
from algorithms.gwo import gwo_auxiliary_functions as gaf
from plot_functions import plot_figma_iter

def run(graph, pop, pop_tam, dim, max_gen, fitness_funtion, lim_min=None, lim_max=None):
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
        fitness[i] = fitness_funtion(graph, pop[i])

    #print(fitness)

    # Determinar lobos alpha, beta e delta
    i_alpha = np.argmax(fitness)
    alpha = pop[i_alpha]

    beta_fitness = np.copy(fitness)
    beta_fitness[i_alpha] = -1
    i_beta = np.argmax(beta_fitness)
    beta = pop[i_beta]

    delta_fitness = np.copy(beta_fitness)
    delta_fitness[i_beta] = -1
    i_delta = np.argmax(delta_fitness)
    delta = pop[i_delta]

    # Plotar posição da primeira população
    #plot_figma_iter(pop[:,0], pop[:,1], alpha, beta, delta, lim_min=lim_min[0][0:2], lim_max=lim_max[0][0:2])

    gen = 1
    while gen < max_gen:
        X1 = gaf.get_x(pop, pop_tam, dim, alpha, big_a_alpha, big_c_alpha)
        X2 = gaf.get_x(pop, pop_tam, dim, beta, big_a_beta, big_c_beta)
        X3 = gaf.get_x(pop, pop_tam, dim, delta, big_a_delta, big_c_delta)

        pop = (X1 + X2 + X3)/3

        ### verificar se os limites não foram estourados
        if lim_max is not None:
            pop[pop > lim_max] = lim_max[pop > lim_max]
        if lim_min is not None:
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
        check_zeros = False
        for i in range(pop_tam):
            try:
                fitness[i] = fitness_funtion(graph, pop[i])
            except:
                print(pop[i])
                check_zeros = True
                break
        
        if check_zeros:
            break

        # Determinar lobos alpha, beta e delta
        i_alpha = np.argmax(fitness)
        alpha = pop[i_alpha]

        beta_fitness = np.copy(fitness)
        beta_fitness[i_alpha] = -1
        i_beta = np.argmax(beta_fitness)
        beta = pop[i_beta]

        delta_fitness = np.copy(beta_fitness)
        delta_fitness[i_beta] = -1
        i_delta = np.argmax(delta_fitness)
        delta = pop[i_delta]

        #if gen % 10 == 0:
            #plot_figma_iter(pop[:,0], pop[:,1], alpha, beta, delta, lim_min=lim_min[0][0:2], lim_max=lim_max[0][0:2])

        print("GEN: {} / RES: {}".format(gen, fitness_funtion(graph, alpha)))
        gen += 1
    
    return alpha