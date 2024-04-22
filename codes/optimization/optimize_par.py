import _pickle as cPickle
import networkx as nx
import numpy as np
from multiprocessing import Pool

def dist_between(node1, node2, x, y):
    return ((x[node1]-x[node2])**2 + (y[node1]-y[node2])**2)**0.5

def calculate_metrics(args):
    node_pair, G, x, y = args
    i, j = node_pair
    # track_length = nx.shortest_path_length(G, source=i, target=j, weight='KM')
    G.add_edge(i, j)
    distance = ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) ** 0.5
    attrs = {(i, j): {"KM": distance}}
    nx.set_edge_attributes(G, attrs)
    # new_efficiency = nx.global_efficiency(G)
    # local_efficiency = nx.local_efficiency(G)
    # cluster_efficiency = nx.average_clustering(G)
    # edge_betweeness = nx.edge_betweenness_centrality_subset(G, i, j)
    ASP = nx.average_shortest_path_length(G, weight='KM')
    G.remove_edge(i, j)
    # distance = dist_between(i, j, x, y)
    return (i, j, distance, ASP)

def print_progress(progress, total):
    percent_complete = (progress / total) * 100
    print(f"Progress: {progress}/{total} ({percent_complete:.2f}%)")




if __name__ == "__main__":
    with open(r"../graphs/graph_fixed_attr.pickle", "rb") as input_file:
        G = cPickle.load(input_file)
    G = nx.Graph(G)
    x = dict(nx.get_node_attributes(G, 'x'))
    y = dict(nx.get_node_attributes(G, 'y'))

    node_pairs = [(i, j) for i in G.nodes for j in G.nodes if j > i]
    edges = set(G.edges)
    node_pairs = [(u, v) for u, v in node_pairs if (u, v) not in edges and (v, u) not in edges]
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
        i, j, distance, ASP = result
        store[(i, j)] = {
            # 'new_efficiency': new_efficiency,
            # 'local_efficiency': local_efficiency,
            # 'cluster_efficiency': cluster_efficiency,
            'node1': i,
            'node2': j,
            # 'track_length': track_length,
            'distance': distance,
            'ASP': ASP
        }

    # Now 'store' contains the computed metrics for each pair of nodes

    with open(r"../files/proposed_ASP.pickle", "wb") as output_file:
        cPickle.dump(store, output_file)

    # Now 'store' contains the computed metrics for each pair of nodes
