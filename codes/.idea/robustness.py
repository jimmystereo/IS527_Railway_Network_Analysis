import copy

import networkx as nx
import _pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt

with open(r"graphs/graph_with_KM.pickle", "rb") as input_file:
    G = cPickle.load(input_file)


## Robustness analysis
# Implement random failure and targeted attack scenarios
# Random failure
lccs = []
fractions = []
for phi in range(0, 101, 5):
    fraction = phi / 100
    G_copy = copy.deepcopy(G)
    nodes_to_remove = [n for n in G_copy.nodes() if np.random.rand() < fraction]
    G_copy.remove_nodes_from(nodes_to_remove)
    try:
        lcc_size = max(len(c) for c in nx.connected_components(G_copy)) / len(G_copy)
    except:
        lcc_size = 0
    fractions.append(1- fraction)
    lccs.append(lcc_size)
plt.plot(fractions, lccs, label = 'Random failure')


# Targeted attack
lccs = []
fractions = []
for phi in range(0, 101, 5):
    fraction = phi / 100
    G_copy = copy.deepcopy(G)
    nodes = list(G_copy.nodes())
    nodes.sort(key=lambda n: G_copy.degree(n), reverse=True)
    nodes_to_remove = nodes[0:int(fraction * len(nodes))]
    G_copy.remove_nodes_from(nodes_to_remove)
    try:
        lcc_size = max(len(c) for c in nx.connected_components(G_copy)) / len(G_copy)
    except:
        lcc_size = 0
    fractions.append(1- fraction)
    lccs.append(lcc_size)


plt.plot(fractions, lccs, label = 'Targeted attack')
plt.xlabel('Fraction of nodes present')
plt.ylabel('Size of the largest cluster S')
plt.title('Robustness of the railway network')
plt.legend()
plt.savefig('plots/robustness.png')