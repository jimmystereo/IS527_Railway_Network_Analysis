import _pickle as cPickle
import networkx as nx
import numpy as np
from multiprocessing import Pool

def dist_between(node1, node2, x, y):
    return ((x[node1]-x[node2])**2 + (y[node1]-y[node2])**2)**0.5

def calculate_metrics(args):
    node_pair, G, x, y = args
    i, j = node_pair
    track_length = nx.shortest_path_length(G, source=i, target=j, weight='KM')
    G.add_edge(i, j)
    new_efficiency = nx.global_efficiency(G)
    local_efficiency = nx.local_efficiency(G)
    cluster_efficiency = nx.average_clustering(G)
    G.remove_edge(i, j)
    distance = dist_between(i, j, x, y)
    return (i, j, new_efficiency, local_efficiency, cluster_efficiency, track_length, distance)

def print_progress(progress, total):
    percent_complete = (progress / total) * 100
    print(f"Progress: {progress}/{total} ({percent_complete:.2f}%)")

if __name__ == "__main__":
    with open(r"../graphs/graph_fixed_attr.pickle", "rb") as input_file:
        G = cPickle.load(input_file)

    x = dict(nx.get_node_attributes(G, 'x'))
    y = dict(nx.get_node_attributes(G, 'y'))

    node_pairs = [(i, j) for i in G.nodes for j in G.nodes if j > i]
    node_pairs = set(node_pairs) - set(G.edges)
    node_pairs = list(node_pairs)
    total_pairs = len(node_pairs)
    num_processes = 6  # Adjust according to the number of CPU cores available
    args_list = [(node_pair, G, x, y) for node_pair in node_pairs]

    with Pool(num_processes) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(calculate_metrics, args_list), 1):
            results.append(result)
            print_progress(i, total_pairs)

    store = {}
    for result in results:
        i, j, new_efficiency, local_efficiency,cluster_efficiency, track_length, distance = result
        store[(i, j)] = {
            'new_efficiency': new_efficiency,
            'local_efficiency': local_efficiency,
            'cluster_efficiency': cluster_efficiency,
            'node1': i,
            'node2': j,
            'track_length': track_length,
            'distance': distance
        }

    # Now 'store' contains the computed metrics for each pair of nodes

    with open(r"../files/proposed_new.pickle", "wb") as output_file:
        cPickle.dump(store, output_file)

    # Now 'store' contains the computed metrics for each pair of nodes
