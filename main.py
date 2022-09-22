import numpy as np
import dataset_loader as dl
import fitness_functions as ff
from algorithms.gwo import gwo

# Definir seed fixa para que os resultados aleatórios sejam sempre os mesmos
np.random.seed(seed=3)

# Carregar o grafo a ser utilizado
graph = dl.karate_club_loader()

# Parametros
pop_tam = 100 # 1000 # tamanho da população
dim = graph.num_nodes # número de entradas do problema
max_gen = 50 # 50 # número max de gerações
limites = [-0.5, (dim+.499)]

# Definição de limites do espaço de busca
lim_min = np.ones(shape=[pop_tam, dim]) * limites[0]
lim_max = np.ones(shape=[pop_tam, dim]) * limites[1]

# População iniciada aleatoriamente
pop = np.random.randint(dim, size=(pop_tam, dim))

# Algoritmo
print("Solução encontrada: {}".format(gwo.run(graph, pop, pop_tam, dim, max_gen, ff.separability, lim_min=lim_min, lim_max=lim_max)))


"""


"""

"""
subset1 = torch.tensor([0, 2, 3, 6], dtype=torch.long)
subset2 = torch.tensor([0, 1, 4, 5, 6], dtype=torch.long)
subgraph1 = graph.subgraph(subset1)
subgraph2 = graph.subgraph(subset2)
"""