import numpy as np
import dataset_loader as dl
import fitness_functions as ff
from algorithms import de, ga, gwo, pso
import pandas as pd
import static_test as st

# Definir seed fixa para que os resultados aleatórios sejam sempre os mesmos
#np.random.seed(seed=3)

# deap
# pygmo

# Carregar o grafo a ser utilizado
graph = dl.karate_club_loader()

# Parametros
pop_tam = 100 # tamanho da população
dim = graph.num_nodes # número de nós do grafo
max_gen = 5 # # número max de gerações
limites = [-0.5, (dim+.499)] # podem existir no máximo 'dim' comunidades, os limites são representados assim

# Definição de limites do espaço de busca
lim_min = np.ones(shape=[pop_tam, dim]) * limites[0]
lim_max = np.ones(shape=[pop_tam, dim]) * limites[1]

# Testes
reps = 3
data = []

for i in range(reps):
    # População iniciada aleatoriamente
    pop = np.random.randint(dim, size=(pop_tam, dim))
    pop = pop.astype(float)

    solucoes = []

    #Algoritmos
    print("Rodada: {}\nGA".format(i+1))
    solucoes.append(ga.run(graph, np.copy(pop), pop_tam, dim, max_gen, ff.modularity))
    print("\nDE")
    solucoes.append(de.run(graph, np.copy(pop), pop_tam, dim, max_gen, ff.modularity, lim_min=lim_min, lim_max=lim_max))
    print("\nPSO")
    solucoes.append(pso.run(graph, np.copy(pop), pop_tam, dim, max_gen, ff.modularity, lim_min=lim_min, lim_max=lim_max))
    print("\nGWO")
    solucoes.append(gwo.run(graph, np.copy(pop), pop_tam, dim, max_gen, ff.modularity, lim_min=lim_min, lim_max=lim_max))
    print()

    scores = []
    for sol in solucoes:
        scores.append(ff.modularity(graph, sol))
    data.append(scores)

columns = ['GA', 'DE', 'PSO', 'GWO']
results = pd.DataFrame(data=data, columns=columns)

print(results)