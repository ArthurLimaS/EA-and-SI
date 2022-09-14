from asyncio.windows_events import NULL
import numpy as np
import matplotlib.pyplot as plt

# função para atualizar a variável 'a'
def func_a(gen, max_gen):
    return 2 * (1 - (gen/max_gen))

def func_big_A(a):
    A = np.zeros(shape=[pop_tam, pop_tam])
    for i in range(len(A)):
        A[i][i] = (2 * a * np.random.rand(1)) - a
    return A

def func_C():
    C = np.zeros(shape=[pop_tam, pop_tam])
    for i in range(len(C)):
        C[i][i] = 2 * np.random.rand(1)
    return C

# função de fitness
def sphere(x):
    return np.sum(np.square(x))

def get_x(pop, figma, A, C):
    x_figma = np.ones(shape=(pop_tam, dim))
    for i in range(len(x_figma)):
        x_figma[i] = figma

    return x_figma - np.dot(A, (np.dot(C, x_figma) - pop))

# função matplot
def plot_figma(x, y, alpha, beta, delta):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set(xlim=(limites[0], limites[1]), xticks = np.arange(-100, 101, 25),
            ylim=(limites[0], limites[1]), yticks = np.arange(-100, 101, 25))
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.grid(True)

    plt.text(x=alpha[0], y=alpha[1], s="Alpha")
    plt.text(x=beta[0], y=beta[1], s="Beta")
    plt.text(x=delta[0], y=delta[1], s="Delta")

    plt.show()

def plot_indexes(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set(xlim=(limites[0], limites[1]), xticks = np.arange(-100, 101, 25),
            ylim=(limites[0], limites[1]), yticks = np.arange(-100, 101, 25))
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.grid(True)

    for i in range(len(x)):
        plt.text(x=x[i], y=y[i], s=i)

    plt.show()


# definir seed fixa para que os resultados aleatórios sejam sempre os mesmos
np.random.seed(seed=3)

# Parametros do algoritmo
pop_tam = 5 # tamanho da população
dim = 2 # número de entradas do problema
max_gen = 100 # número max de gerações
limites = [-100, 100] # limites inferiores e superiores

# População iniciada aleatoriamente
lim_min = limites[0] * np.ones(shape=[pop_tam, dim])
lim_max = limites[1] * np.ones(shape=[pop_tam, dim])

pop = np.random.rand(pop_tam, dim)
pop = pop * (lim_max - lim_min) + lim_min

print(pop)
plot_indexes(pop[:,0], pop[:,1])

# Inicializar variáveis a, A e C
a = func_a(0, max_gen)
A = func_big_A(a)
C = func_C()

# Calculo de fitness da população inicial
fitness = np.zeros(shape=(pop_tam))

for i in range(pop_tam):
    fitness[i] = sphere(pop[i])

print(fitness)

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
plot_figma(pop[:,0], pop[:,1], alpha, beta, delta)

gen = 1
while gen < max_gen:
    X1 = get_x(pop, alpha, A, C)
    X2 = get_x(pop, beta, A, C)
    X3 = get_x(pop, delta, A, C)

    pop = (X1 + X2 + X3)/3

    ### verificar se os limites não foram estourados

    a = func_a(gen, max_gen)
    A = func_big_A(a)
    C = func_C()

    # Calculo de fitness
    for i in range(pop_tam):
        fitness[i] = sphere(pop[i])

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

    #print("GEN: {} / RES: {}".format(gen, sphere(alpha)))
    gen += 1
