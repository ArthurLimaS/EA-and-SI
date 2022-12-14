import numpy as np


def roullete_selector(n, pop, fitness, max, tolerance=(3.4 * (10 ** -4))):
    sel_pop = np.zeros(shape=(n, len(pop[0])))
    sel_pop.astype(float)

    # Definir probabilidades
    if max:
        probabilities = fitness/(np.sum(fitness) + tolerance)
        probabilities[-1] += tolerance/(np.sum(fitness) + tolerance)
        probabilities = np.cumsum(probabilities)
    else:
        probabilities = np.sum(fitness)/(fitness + tolerance)
        probabilities[-1] += tolerance/(np.sum(fitness) + tolerance)
        probabilities = probabilities/(np.sum(probabilities))
        probabilities = np.cumsum(probabilities)

    # Selecionar n individuos
    cnt_sel = 0
    while cnt_sel < n:
        r1 = np.random.rand()
        indices1 = probabilities >= r1
        s1 = np.argmax(indices1)

        iguais = True
        s2 = -1
        while iguais:
            r2 = np.random.rand()
            indices2 = probabilities >= r2
            s2 = np.argmax(indices2)
            if s1 != s2:
                iguais = False

        sel_pop[cnt_sel,:] = pop[s1, :]
        cnt_sel += 1
        sel_pop[cnt_sel,:] = pop[s2, :]
        cnt_sel += 1

    return sel_pop


def crossover(pop_tam, dim, parents, cr, n_pop):
    cnt_crs = 0
    while cnt_crs < pop_tam:
        n_p1 = np.zeros(shape=[dim])
        n_p1.astype(float)
        n_p2 = np.zeros(shape=[dim])
        n_p2.astype(float)

        p1 = np.copy(parents[cnt_crs])
        p2 = np.copy(parents[cnt_crs+1])

        r1 = np.random.rand()
        if r1 <= cr:
            n_p1 = p1
            n_p2 = p2

            cxpoint = np.random.randint(1, len(n_p1) - 1)
            temp_ind = np.copy(n_p1)
            n_p1[cxpoint:] = n_p2[cxpoint:]
            n_p2[cxpoint:] = temp_ind[cxpoint:]

            n_pop[cnt_crs,:] = n_p1
            cnt_crs += 1
            n_pop[cnt_crs,:] = n_p2
            cnt_crs += 1
        else:
            n_pop[cnt_crs,:] = p1
            cnt_crs += 1
            n_pop[cnt_crs,:] = p2
            cnt_crs += 1


def mutation(pop_tam, dim, mr, n_pop, comms_count):
    for i in range(pop_tam):
        for j in range(dim):
            r = np.random.rand()
            if r <= mr:
                n_pop[i][j] = np.random.randint(0, comms_count)


def run(graph, pop_ini, pop_tam, dim, max_gen, fitness_function, comms_count, max=True, cr=0.9, mr=0.01, rep=0):
    # Gr??fico
    y = []

    pop = np.copy(pop_ini)

    # 2. Avaliar cada indiv??duo da popula????o.
    fitness = np.zeros(shape=[pop_tam])
    fitness.astype(float)

    for i in range(pop_tam):
        fitness[i] = fitness_function(graph, pop[i])

    best_index = 0
    if max:
        best_index = np.argmax(fitness)
    else:
        best_index = np.argmin(fitness)

    # 3. Enquanto crit??rio de parada n??o for satisfeito fa??a
    gen = 1
    while gen <= max_gen:
        old_pop = pop.copy()

        n_pop = np.zeros(shape=(pop_tam, dim))
        n_pop.astype(float)

        # 3.1 Selecionar os indiv??duos mais aptos.
        parents = roullete_selector(pop_tam, old_pop, fitness, max)

        # 3.2 Criar novos indiv??duos aplicando os operadores crossover e muta????o.
        # 3.2.1 Crossover
        crossover(pop_tam, dim, parents, cr, n_pop)

        # 3.2.2 Mutation
        mutation(pop_tam, dim, mr, n_pop, comms_count)
        
        # 3.3 Armazenar os novos indiv??duos em uma nova popula????o
        pop = np.copy(n_pop)

        # 3.4 Avaliar cada cromossomo da nova popula????o.
        for i in range(pop_tam):
            fitness[i] = fitness_function(graph, pop[i])

        if max:
            best_index = np.argmax(fitness)
        else:
            best_index = np.argmin(fitness)
        
        y.append(fitness_function(graph, pop[best_index]))
        print("GEN: {} / RES: {}".format(gen, fitness_function(graph, pop[best_index])))
        gen += 1

    #x = [i+1 for i in range(max_gen)]
    #plt.plot(x, y)
    #plt.title("GA")
    #plt.ylabel("modularity value")
    #plt.ylim(0, 0.5)
    #plt.xlabel("generation")
    #plt.savefig("graphs/GA_{}.png".format(rep+1))
    #plt.close()

    
    return pop[best_index], y