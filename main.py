import numpy as np
import dataset_loader as dl
import fitness_functions as ff
from algorithms import de, ga, gwo, pso, am
import pandas as pd
import matplotlib.pyplot as plt

# Definir seed fixa para que os resultados aleatórios sejam sempre os mesmos
#np.random.seed(seed=3)

# deap
# pygmo

# Carregar o grafo a ser utilizado
graph = dl.karate_club_loader()

# Parametros
pop_tam = 1000 # tamanho da população
dim = graph.num_nodes # número de nós do grafo
comms_count = 2
max_gen = 50 # número max de gerações
limites = [-0.5, ((comms_count-1)+.499)] # podem existir no máximo 'dim' comunidades, os limites são representados assim

# Definição de limites do espaço de busca
lim_min = np.ones(shape=[pop_tam, dim]) * limites[0]
lim_max = np.ones(shape=[pop_tam, dim]) * limites[1]

# Testes
reps = 10
ga_data = np.zeros(shape=[reps, max_gen])
de_data = np.zeros(shape=[reps, max_gen])
pso_data = np.zeros(shape=[reps, max_gen])
gwo_data = np.zeros(shape=[reps, max_gen])
am_data = np.zeros(shape=[reps, max_gen])
modularity_data = []
comms_data = []

for i in range(reps):
    # População iniciada aleatoriamente
    pop = np.random.randint(comms_count, size=(pop_tam, dim))
    pop = pop.astype(float)

    solucoes = []

    #Algoritmos
    print("Rodada: {}\nGA".format(i+1))
    solucoes.append(ga.run(graph, np.copy(pop), pop_tam, dim, max_gen, ff.modularity, comms_count, rep=i))
    print("\nDE")
    solucoes.append(de.run(graph, np.copy(pop), pop_tam, dim, max_gen, ff.modularity, lim_min=lim_min, lim_max=lim_max, rep=i))
    print("\nPSO")
    solucoes.append(pso.run(graph, np.copy(pop), pop_tam, dim, max_gen, ff.modularity, lim_min=lim_min, lim_max=lim_max, rep=i))
    print("\nGWO")
    solucoes.append(gwo.run(graph, np.copy(pop), pop_tam, dim, max_gen, ff.modularity, lim_min=lim_min, lim_max=lim_max, rep=i))
    print("\nAM")
    solucoes.append(am.run(graph, np.copy(pop), pop_tam, dim, max_gen, ff.modularity, comms_count, rep=i))
    print()

    scores = []
    comms = []
    c = 0
    for sol in solucoes:
        scores.append(ff.modularity(graph, sol[0]))

        for j in range(len(sol[0])):   # arrendondar os valores do individuo (para o caso do algoritmo trabalhar com floats ao invés de inteiros)
            sol[0][j] = round(sol[0][j])
        comms.append(sol[0])

        if c == 0:
            ga_data[i] = sol[1]
        if c == 1:
            de_data[i] = sol[1]
        if c == 2:
            pso_data[i] = sol[1]
        if c == 3:
            gwo_data[i] = sol[1]
        if c == 4:
            am_data[i] = sol[1]
        
        c += 1

    modularity_data.append(scores)
    comms_data.append(comms)

pd.options.display.float_format = "{:,.4f}".format
columns = ['GA', 'DE', 'PSO', 'GWO', 'AM']

modularity_results = pd.DataFrame(data=modularity_data, columns=columns)
comms_results = pd.DataFrame(data=comms_data, columns=columns)

print(modularity_results)
modularity_results.to_csv('modularity_results.csv', index=False)
#print(comms_results)
comms_results.to_csv('comms_results.csv', index=False)

x = [i+1 for i in range(max_gen)]

# GA
y = []
for i in range(len(ga_data[0])):
    gen_y = ga_data[:,i]
    media = np.mean(np.array(gen_y))
    y.append(media)
plt.plot(x, y)
plt.title("GA")
plt.ylabel("mean modularity value")
plt.ylim(0, 0.4)
plt.xlabel("generation")
plt.savefig("graphs/GA.png")
plt.close()

# DE
y = []
for i in range(len(de_data[0])):
    gen_y = de_data[:,i]
    media = np.mean(np.array(gen_y))
    y.append(media)
plt.plot(x, y)
plt.title("DE")
plt.ylabel("mean modularity value")
plt.ylim(0, 0.4)
plt.xlabel("generation")
plt.savefig("graphs/DE.png")
plt.close()

# PSO
y = []
for i in range(len(pso_data[0])):
    gen_y = pso_data[:,i]
    media = np.mean(np.array(gen_y))
    y.append(media)
plt.plot(x, y)
plt.title("PSO")
plt.ylabel("mean modularity value")
plt.ylim(0, 0.4)
plt.xlabel("generation")
plt.savefig("graphs/PSO.png")
plt.close()

# GWO
y = []
for i in range(len(gwo_data[0])):
    gen_y = gwo_data[:,i]
    media = np.mean(np.array(gen_y))
    y.append(media)
plt.plot(x, y)
plt.title("GWO")
plt.ylabel("mean modularity value")
plt.ylim(0, 0.4)
plt.xlabel("generation")
plt.savefig("graphs/GWO.png")
plt.close()

# AM
y = []
for i in range(len(am_data[0])):
    gen_y = am_data[:,i]
    media = np.mean(np.array(gen_y))
    y.append(media)
plt.plot(x, y)
plt.title("AM")
plt.ylabel("mean modularity value")
plt.ylim(0, 0.4)
plt.xlabel("generation")
plt.savefig("graphs/AM.png")
plt.close()