from re import sub
import numpy as np
import torch


def sphere(x):
    return np.sum(np.square(x))


def separability(graph, ind, verbose=False):
    r_ind = np.round(ind) # arrendondar os valores do individuo (para o caso do algoritmo trabalhar com floats ao invés de inteiros)
    comms = np.unique(r_ind) # comunidades definidas

    if len(comms) == 1:
        return 0

    sep = 0

    for comm in comms:
        subset = np.where(r_ind == comm)
        tensorsubset = torch.tensor(subset[0], dtype=torch.long)
        subgraph = graph.subgraph(tensorsubset)

        if verbose:
            print("Subgraph: {}".format(subgraph.x))
        internal_edges = 0
        external_edges = 0
        for i in range(len(graph.edge_index[0])):
            edge = graph.edge_index[:,i]

            if (edge[0] in subgraph.x) \
                and (edge[1] in subgraph.x):
                internal_edges += 1
                
                #if verbose:
                    #print("Edge v in C: {}".format(edge))
            
            if (edge[0] in subgraph.x) \
                and (edge[1] not in subgraph.x):
                external_edges += 1

                #if verbose:
                    #print("Edge v not in C: {}".format(edge))
        
        sep += internal_edges / external_edges
    
    return sep/len(comms)


def test_input():
    import dataset_loader as dl
    graph = dl.karate_club_loader()
    print(separability(graph, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0], verbose=True))