import matplotlib.pyplot as plt
import numpy as np

def plot_figma(x, y, alpha, beta, delta, limites):
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

#plot_indexes(pop[:,0], pop[:,1])
def plot_indexes(x, y, limites):
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

#plot_figma_iter(pop[:,0], pop[:,1], alpha, beta, delta)
def plot_figma_iter(x, y, alpha, beta, delta):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set(xlim=(np.min(x)-np.abs(np.min(x)), np.max(x)+np.abs(np.max(x))),
            ylim=(np.min(y)-np.abs(np.min(y)), np.max(y)+np.abs(np.max(x))))
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.grid(True)

    plt.text(x=alpha[0], y=alpha[1], s="Alpha")
    plt.text(x=beta[0], y=beta[1], s="Beta")
    plt.text(x=delta[0], y=delta[1], s="Delta")

    plt.show()