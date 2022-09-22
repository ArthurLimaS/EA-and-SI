import numpy as np


def roullete_selector(n, pop, fitness, max, tolerance=(3.4 * (10 ** -4))):
  sel_pop = np.zeros(shape=(n, len(pop[0])))
  
  # Definir probabilidades
  probabilities = np.zeros(shape=fitness.shape)

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
    r1 = np.random.rand(1)
    indices1 = probabilities >= r1
    s1 = np.argmax(indices1)
    
    iguais = True
    s2 = -1
    while iguais:
      r2 = np.random.rand(1)
      indices2 = probabilities >= r2
      s2 = np.argmax(indices2)
      if s1 != s2:
        iguais = False

    sel_pop[cnt_sel,:] = pop[s1, :]
    cnt_sel += 1
    sel_pop[cnt_sel,:] = pop[s2, :]
    cnt_sel += 1
  
  return sel_pop


def run(graph, pop_ini, pop_tam, dim, max_gen, fitness_function, max=True, cr=0.9, mr=0.01):
    pop = np.copy(pop_ini)

    # 2. Avaliar cada indivíduo da população.
    fitness = np.zeros(shape=[pop_tam])

    for i in range(pop_tam):
        fitness[i] = fitness_function(graph, pop[i,:])

    index = 0
    if max:
        index = np.argmax(fitness)
    else:
        index = np.argmin(fitness)

    # 3. Enquanto critério de parada não for satisfeito faça
    gen = 1
    while gen <= max_gen:
        n_pop = np.zeros(shape=(pop_tam, dim))
        # 3.1 Selecionar os indivíduos mais aptos.
        parents = roullete_selector(pop_tam, pop, fitness, max)

        # 3.2 Criar novos indivíduos aplicando os operadores crossover e mutação.
        # 3.2.1 Crossover
        cnt_crs = 0
        while cnt_crs < pop_tam:
            n_p1 = np.zeros(shape=[dim])
            n_p2 = np.zeros(shape=[dim])

            p1 = parents[cnt_crs]
            p2 = parents[cnt_crs+1]

            r = np.random.rand(1)
            if r <= cr:
                # Cross
                r = np.random.randint(0,dim)
                indexes = np.arange(dim)
                n_p1 = p1 * (indexes < r) + p2 * (indexes >= r)
                n_p2 = p1 * (indexes >= r) + p2 * (indexes < r)

            n_pop[cnt_crs,:] = n_p1
            cnt_crs += 1
            n_pop[cnt_crs,:] = n_p2
            cnt_crs += 1

        # 3.2.2 Mutation
        for i in range(pop_tam):
            for j in range(dim):
                r = np.random.rand(1)
                if r <= mr:
                    n_pop[i][j] = np.random.randint(dim)
        # 3.3 Armazenar os novos indivíduos em uma nova população
        pop = n_pop

        # 3.4 Avaliar cada cromossomo da nova população.
        for i in range(pop_tam):
            fitness[i] = fitness_function(graph, pop[i,:])

        if max:
            index = np.argmax(fitness)
        else:
            index = np.argmin(fitness)
        
        print("GEN: {} / RES: {}".format(gen, fitness_function(graph, pop[index])))
        gen += 1
    
    return pop[index]