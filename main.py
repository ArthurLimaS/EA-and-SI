import numpy as np
import algorithms as algs
import fitness_functions as ff
import plot_functions
import dataset_loader

#dataset_loader.dblp_loader()

# definir seed fixa para que os resultados aleatórios sejam sempre os mesmos
np.random.seed(seed=3)

# Parametros
pop_tam = 5 # tamanho da população
dim = 2 # número de entradas do problema
max_gen = 100 # número max de gerações
limites = [-100, 100] # limites inferiores e superiores

# População iniciada aleatoriamente
lim_min = limites[0] * np.ones(shape=[pop_tam, dim])
lim_max = limites[1] * np.ones(shape=[pop_tam, dim])
pop = np.random.rand(pop_tam, dim)
pop = pop * (lim_max - lim_min) + lim_min