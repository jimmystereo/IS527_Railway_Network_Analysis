import _pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy
from sklearn import metrics
with open(r"graphs/graph_fixed_attr.pickle", "rb") as input_file:
    G = cPickle.load(input_file)
with open(r"graphs/new_track.pickle", "rb") as input_file:
    G2 = cPickle.load(input_file)
def simulate(G):
    result = {
        'Lcc':{'target': 0, 'random': 0},
        'efficiency':{'target': 0, 'random': 0}
    }
    # Implement random failure and targeted attack scenarios
    # Random failure
    lccs = []
    fractions = []
    # [len(c) for c in nx.connected_components(G_copy)]
    for phi in range(0, 101, 5):
        fraction = phi / 100
        G_copy = copy.deepcopy(G)
        nodes_to_remove = [n for n in G_copy.nodes() if np.random.rand() < fraction]
        G_copy.remove_nodes_from(nodes_to_remove)
        try:
            lcc_size = max(len(c) for c in nx.connected_components(G_copy)) / len(G_copy)
            # Plot lcc_size vs. fraction of nodes present
            fractions.append(fraction)
            lccs.append(lcc_size)
        except:
            continue
    result['Lcc']['random'] =  metrics.auc(fractions, lccs)
    # plt.plot(fractions, lccs, label="Random Failure")

    lccs = []
    fractions = []
    # Targeted attack
    for phi in range(0, 101, 5):
        fraction = phi / 100
        G_copy = copy.deepcopy(G)
        nodes = list(G_copy.nodes())
        nodes.sort(key=lambda n: G_copy.degree(n), reverse=True)
        nodes_to_remove = nodes[:int(fraction * len(nodes))]
        G_copy.remove_nodes_from(nodes_to_remove)
        try:
            lcc_size = max(len(c) for c in nx.connected_components(G_copy)) / len(G_copy)
            # Plot lcc_size vs. fraction of nodes present
            fractions.append(fraction)
            lccs.append(lcc_size)
        except:
            continue
    result['Lcc']['target'] =  metrics.auc(fractions, lccs)


    lccs = []
    fractions = []

    # Implement random failure and targeted attack scenarios
    # Random failure
    lccs = []
    fractions = []
    # [len(c) for c in nx.connected_components(G_copy)]
    for phi in range(0, 101, 5):
        fraction = phi / 100
        G_copy = copy.deepcopy(G)
        nodes_to_remove = [n for n in G_copy.nodes() if np.random.rand() < fraction]
        G_copy.remove_nodes_from(nodes_to_remove)
        try:
            lcc_size = nx.global_efficiency(G_copy)
            # Plot lcc_size vs. fraction of nodes present
            fractions.append(fraction)
            lccs.append(lcc_size)
        except:
            continue

    lccs = []
    fractions = []

    # Targeted attack
    for phi in range(0, 101, 5):
        fraction = phi / 100
        G_copy = copy.deepcopy(G)
        nodes = list(G_copy.nodes())
        nodes.sort(key=lambda n: G_copy.degree(n), reverse=True)
        nodes_to_remove = nodes[:int(fraction * len(nodes))]
        G_copy.remove_nodes_from(nodes_to_remove)
        try:
            lcc_size = nx.global_efficiency(G_copy)
            fractions.append(fraction)
            lccs.append(lcc_size)
        except:
            continue
    result['efficiency']['random'] =  metrics.auc(fractions, lccs)


    lccs = []
    fractions = []

    # Targeted attack - G2
    for phi in range(0, 101, 5):
        fraction = phi / 100
        G_copy = copy.deepcopy(G2)
        nodes = list(G_copy.nodes())
        nodes.sort(key=lambda n: G_copy.degree(n), reverse=True)
        nodes_to_remove = nodes[:int(fraction * len(nodes))]
        G_copy.remove_nodes_from(nodes_to_remove)
        try:
            lcc_size = nx.global_efficiency(G_copy)
            fractions.append(fraction)
            lccs.append(lcc_size)
        except:
            continue
    result['efficiency']['target'] =  metrics.auc(fractions, lccs)
    return result
simulate(G)
simulate(G2)