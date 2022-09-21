import numpy as np

def func_a(gen, max_gen):
    return 2 * (1 - (gen/max_gen))

def func_big_a(a, pop_tam):
    A = np.zeros(shape=[pop_tam, pop_tam])
    for i in range(len(A)):
        A[i][i] = (2 * a * np.random.rand(1)) - a
    return A

def func_big_c(pop_tam):
    C = np.zeros(shape=[pop_tam, pop_tam])
    for i in range(len(C)):
        C[i][i] = 2 * np.random.rand(1)
    return C

def get_x(pop, pop_tam, dim, figma, big_a, big_c):
    x_figma = np.ones(shape=(pop_tam, dim))
    for i in range(len(x_figma)):
        x_figma[i] = figma

    return x_figma - np.dot(big_a, (np.dot(big_c, x_figma) - pop))
