import numpy as np
import dataset_loader as dl
import fitness_functions as ff
from algorithms import de, ga, gwo, pso

# Definir seed fixa para que os resultados aleatórios sejam sempre os mesmos
np.random.seed(seed=3)

# Carregar o grafo a ser utilizado
graph = dl.karate_club_loader()

# Parametros
pop_tam = 1000 # tamanho da população
dim = graph.num_nodes # número de nós do grafo
max_gen = 50 # # número max de gerações
limites = [-0.5, (dim+.499)] # podem existir no máximo 'dim' comunidades, os limites são representados assim

# Definição de limites do espaço de busca
lim_min = np.ones(shape=[pop_tam, dim]) * limites[0]
lim_max = np.ones(shape=[pop_tam, dim]) * limites[1]

# População iniciada aleatoriamente
pop = np.random.randint(dim, size=(pop_tam, dim))
pop = pop.astype(float)

# Algoritmo
solucoes = []

print("GWO")
solucoes.append(gwo.run(graph, np.copy(pop), pop_tam, dim, max_gen, ff.modularity, lim_min=lim_min, lim_max=lim_max))
print("PSO")
solucoes.append(pso.run(graph, np.copy(pop), pop_tam, dim, max_gen, ff.modularity, lim_min=lim_min, lim_max=lim_max))
print("GA")
solucoes.append(ga.run(graph, np.copy(pop), pop_tam, dim, max_gen, ff.modularity))
print("DE")
solucoes.append(de.run(graph, np.copy(pop), pop_tam, dim, max_gen, ff.modularity))

print("Solução encontrada GWO: {}".format(solucoes[0]))
print("Solução encontrada PSO: {}".format(solucoes[1]))
print("Solução encontrada GA: {}".format(solucoes[2]))
print("Solução encontrada DE: {}".format(solucoes[3]))