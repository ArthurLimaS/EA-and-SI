import torch
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub
import numpy as np


def dblp_loader():
    dblp = open("dblp.txt")
    dblp.readline() # pular primeira linha

    for line in dblp:
        if line != '':
            prefix = line[0:2]
        else:
            prefix = ''

        if prefix == '#*': # paperTitle
            print("1")
        elif prefix == '#@': # Authors
            print("2")
        elif prefix == '#t': # Year
            print("3")
        elif prefix == '#c': # publication venue
            print("4")
        elif prefix == '#index': # index id of this paper
            print("5")
        elif prefix == '#%': # the id of references of this paper (there are multiple lines, with each indicating a reference)
            print("6")
        elif prefix == '#!': # abstract
            print("7")

    """
    count = 0
    for x in file:
        print(x[0:2], end="")
        count += 1

        if count == 10:
            break
    """


def test_function():
    edge_index = torch.tensor([[3, 6, 3, 0, 3, 2, 2, 6, 0, 6, 0, 5, 5, 6, 5, 4, 5, 1],
                               [6, 3, 0, 3, 2, 3, 6, 2, 6, 0, 5, 0, 6, 5, 4, 5, 1, 5]], dtype=torch.long)
    x = torch.tensor([[0], [1], [2], [3], [4], [5], [6]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    return data


def get_adj_matrix(x, edge_index):
    adj_matrix = np.zeros(shape=[len(x), len(x)])
    adj_matrix.astype(float)

    for i in range(len(x)):
        for j in range(len(x)):
            for index in range(len(edge_index[0])):
                edge = edge_index[:,index]

                if (edge[0] == i) \
                    and (edge[1] == j):
                    adj_matrix[i][j] = 1
                    break
    
    return adj_matrix


def get_k_matrix(x, edge_index):
    k_matrix = np.zeros(shape=[len(x), len(x)])
    k_matrix.astype(float)

    for i in range(len(x)):
        for j in range(len(x)):
            ki = 0
            kj = 0
            for edge in edge_index[0]:
                if edge == i:
                    ki += 1
                if edge == j:
                    kj += 1
            
            k_matrix[i][j] = (ki*kj)/(2*len(edge_index[0]))
    
    return k_matrix


def karate_club_loader():
    temp_data = KarateClub().data
    edge_index = temp_data.edge_index

    x_aux = [[x] for x in range(len(temp_data.x))]
    x = torch.tensor(x_aux, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)

    adj_matrix = get_adj_matrix(x, edge_index)
    
    k_matrix = get_k_matrix(x, edge_index)

    graph = Graph(data, adj_matrix, k_matrix)
    return graph


class Graph():
    def __init__(self, graph, adj_matrix, k_matrix):
        self.x = graph.x
        self.edge_index = graph.edge_index
        self.adj_matrix = adj_matrix
        self.k_matrix = k_matrix
        self.num_nodes = graph.num_nodes