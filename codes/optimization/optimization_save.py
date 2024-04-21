import _pickle as cPickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time

with open(r"../graphs/graph_fixed_attr.pickle", "rb") as input_file:
    G = cPickle.load(input_file)
len(G.nodes)
nx.global_efficiency(G)
G.nodes

G.add_edge(404059, 389582)

def dist_between(node1, node2):
    return ((x[node1]-x[node2])**2+(y[node1]-y[node2])**2)**(0.5)

best_combo = {
    "delta_efficiency": 0,
    "coefficient":0,
    "track_length": None,
    "node1": None,
    "node2": None,
}
x = dict(nx.get_node_attributes(G, 'x'))
y = dict(nx.get_node_attributes(G, 'y'))
ridership = dict(nx.get_node_attributes(G, 'ridership'))
efficiency = nx.global_efficiency(G)

store = {}
for i in G.nodes:
    for j in G.nodes:
        if j <= i:
            continue
        if i!=j and (i, j) not in G.edges:
            track_length =  nx.shortest_path_length(G, source=i, target=j, weight='KM')
            G.add_edge(i,j)
            new_efficiency = nx.global_efficiency(G)
            # delta_efficiency = new_efficiency - efficiency
            distance = dist_between(i, j)
            # coefficient = delta_efficiency*ridership[i]*ridership[j]*track_length/distance
            G.remove_edge(i,j)
            store[(i,j)] = {}
            store[(i,j)]['new_efficiency'] = new_efficiency
            # store[(i,j)]['delta_coefficieny'] = delta_efficiency
            store[(i,j)]['node1'] = i
            store[(i,j)]['node2'] = j
            store[(i,j)]['track_length'] = track_length
            # store[(i,j)]['coefficient'] = coefficient
            store[(i,j)]['distance'] = dist_between(i, j)


