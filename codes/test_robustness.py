import _pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy

with open(r"graphs/graph_fixed_attr.pickle", "rb") as input_file:
    G = cPickle.load(input_file)
with open(r"graphs/greedy/new_track_step_9.pickle", "rb") as input_file:
    G2 = cPickle.load(input_file)

# Targeted attack - G
lccs = []
fractions = []
for phi in range(0, 101, 5):
    fraction = phi / 100
    G_copy = copy.deepcopy(G)
    nodes = list(G_copy.nodes())
    nodes.sort(key=lambda n: G_copy.nodes[n]['ridership'], reverse=True)
    nodes_to_remove = nodes[:int(fraction * len(nodes))]
    G_copy.remove_nodes_from(nodes_to_remove)
    try:
        lcc_size = max(len(c) for c in nx.connected_components(G_copy)) / len(G_copy)
        fractions.append(fraction)
        lccs.append(lcc_size)
    except:
        continue
plt.plot(fractions, lccs, label="Targeted Attack - G")

# Targeted attack - G2
lccs = []
fractions = []
for phi in range(0, 101, 5):
    fraction = phi / 100
    G_copy = copy.deepcopy(G2)
    nodes = list(G_copy.nodes())
    nodes.sort(key=lambda n: G_copy.nodes[n]['ridership'], reverse=True)
    nodes_to_remove = nodes[:int(fraction * len(nodes))]
    G_copy.remove_nodes_from(nodes_to_remove)
    try:
        lcc_size = max(len(c) for c in nx.connected_components(G_copy)) / len(G_copy)
        fractions.append(fraction)
        lccs.append(lcc_size)
    except:
        continue
plt.plot(fractions, lccs, label="Targeted Attack - G2")

plt.legend()
plt.xlabel("Fraction of Failure Nodes")
plt.ylabel("Size of the largest cluster S")
plt.title("Lcc after ridership target attack")
plt.savefig("plots/robustness/robustness_ridership_lcc.png")
plt.show()


# Targeted attack - G
lccs = []
fractions = []
for phi in range(0, 101, 5):
    fraction = phi / 100
    G_copy = copy.deepcopy(G)
    nodes = list(G_copy.nodes())
    nodes.sort(key=lambda n: G_copy.nodes[n]['ridership'], reverse=True)
    nodes_to_remove = nodes[:int(fraction * len(nodes))]
    G_copy.remove_nodes_from(nodes_to_remove)
    try:
        lcc_size = nx.global_efficiency(G_copy)
        fractions.append(fraction)
        lccs.append(lcc_size)
    except:
        continue
plt.plot(fractions, lccs, label="Targeted Attack - G")

# Targeted attack - G2
lccs = []
fractions = []
for phi in range(0, 101, 5):
    fraction = phi / 100
    G_copy = copy.deepcopy(G2)
    nodes = list(G_copy.nodes())
    nodes.sort(key=lambda n: G_copy.nodes[n]['ridership'], reverse=True)
    nodes_to_remove = nodes[:int(fraction * len(nodes))]
    G_copy.remove_nodes_from(nodes_to_remove)
    try:
        lcc_size = nx.global_efficiency(G_copy)
        fractions.append(fraction)
        lccs.append(lcc_size)
    except:
        continue
plt.plot(fractions, lccs, label="Targeted Attack - G2")
plt.legend()
plt.xlabel("Fraction of Failure Nodes")
plt.ylabel("Global Efficiency")
plt.title("Global Efficiency after ridership target attack")
plt.savefig("plots/robustness/robustness_ridership_efficiency.png")
plt.show()