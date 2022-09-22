from tokenize import String
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub

def dblp_loader():
    dblp = open("dblp.txt")
    dblp.readline() # pular primeira linha

    for line in dblp:
        if line != '':
            prefix = line[0:2]
        else:
            prefix = ''

        if prefix == '#*': # paperTitle
            a = 0
        elif prefix == '#@': # Authors
            a = 0
        elif prefix == '#t': # Year
            a = 0
        elif prefix == '#c': # publication venue
            a = 0
        elif prefix == '#index': # index id of this paper
            a = 0
        elif prefix == '#%': # the id of references of this paper (there are multiple lines, with each indicating a reference)
            a = 0
        elif prefix == '#!': # abstract
            a = 0

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


def karate_club_loader():
    temp_data = KarateClub().data
    edge_index = temp_data.edge_index

    x_aux = [[x] for x in range(len(temp_data.x))]
    x = torch.tensor(x_aux, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)
    return data


"""
print(karate_club_loader())
print(test_function())
"""