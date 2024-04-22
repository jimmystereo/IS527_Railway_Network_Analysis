import _pickle as cPickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
import pandas as pd
import copy
import statistics
import math
import _pickle as cPickle
import networkx as nx
import numpy as np
from multiprocessing import Pool


def dist_between(node1, node2, x, y):
    return math.sqrt((x[node1] - x[node2]) ** 2 + (y[node1] - y[node2]) ** 2)


def standardize(x, mean, std):
    return (x - mean) / std


def normalize(x, max, min):
    min = 0.99 * min
    return (x - min) / (max - min)


def calculate_metrics(args):
    node_pair, G, x, y = args
    i, j = node_pair
    track_length = nx.shortest_path_length(G, source=i, target=j, weight='KM')
    G.add_edge(i, j)
    new_efficiency = nx.global_efficiency(G)
    # local_efficiency = nx.local_efficiency(G)
    # cluster_efficiency = nx.average_clustering(G)
    G.remove_edge(i, j)
    distance = dist_between(i, j, x, y)
    return (i, j, new_efficiency,distance)


def print_progress(progress, total):
    percent_complete = (progress / total) * 100
    print(f"Progress: {progress}/{total} ({percent_complete:.2f}%)")


def add_track(store, Graph, step, top_edges):
    G = copy.deepcopy(Graph).to_undirected()

    def normalize(x, max, min):
        min = 0.99 * min
        return (x - min) / (max - min)

    normalized_store = copy.deepcopy(store)
    for key in ['new_efficiency','distance']:
        max_num = max([item[key] for item in store.values()])
        min_num = min([item[key] for item in store.values()])
        for i in normalized_store.keys():
            normalized_store[i][key] = normalize(normalized_store[i][key], max_num, min_num)

    x = dict(nx.get_node_attributes(G, 'x'))
    y = dict(nx.get_node_attributes(G, 'y'))
    ridership = dict(nx.get_node_attributes(G, 'ridership'))
    max_num = max(ridership.values())
    min_num = min(ridership.values())
    for i in G.nodes:
        ridership[i] = normalize(ridership[i], max_num, min_num)

    neighbors = {}
    for n in G.nodes():
        neighbors[n] = list(G.neighbors(n))
    avg_neighbor_ridership = {}
    for n in G.nodes():
        ridership_n = 0
        for n2 in neighbors[n]:
            ridership_n += ridership[n2] / len(neighbors[n])
        avg_neighbor_ridership[n] = ridership_n
    beta = 1
    alpha = 1
    for i, j in normalized_store.keys():
        distance = normalized_store[(i, j)]["distance"]
        coefficient = (store[(i,j)]["new_efficiency"]**2 - (ridership[i]*ridership[j]))/distance
        normalized_store[(i, j)]['coefficient'] = coefficient

    # Print or use sorted_store as needed
    df = pd.DataFrame(normalized_store).T
    df = df.reset_index(drop=True)

    G2 = copy.deepcopy(G)
    i = -1
    c = 1
    highbar = df['distance'].quantile(.05)
    # highbar = normalized_store(3.669742715254712, max([item['distance'] for item in store.values()]),min([item['distance'] for item in store.values()]))
    low = df['distance'].quantile(.01)
    # df = df[df['distance'] <=  highbar]
    # df = df[df['distance'] >=  low]
    # df = df[df['track_length'] < df['track_length'].quantile(.2)]
    top_df = df.sort_values(by='coefficient', ascending=False).reset_index(drop=True)
    added_edges = []

    while c <= 1:
        i += 1
        tops = top_df.loc[i, ['node1', 'node2']]
        node1 = int(tops["node1"])
        node2 = int(tops["node2"])
        if (node1, node2) in G2.edges:
            print('redundant')
            continue
        G2.add_edge(node1, node2)
        added_edges.append((node1, node2))
        c += 1

    # Define positions, edge labels, and node labels
    pos = {}
    # KM = nx.get_edge_attributes(G2, 'KM')
    # name = nx.get_node_attributes(G2, 'name')
    # KM = {i: round(KM[i], 2) for i in KM}
    for i in G2.nodes:
        pos[i] = (x[i], y[i])
    # labels = {node: name[node] for node in G2.nodes}

    # Draw the network with different edge colors
    plt.figure(figsize=(150, 90))
    for edge in G2.edges():
        if (edge[0], edge[1]) in top_edges or (edge[1], edge[0]) in top_edges:  # Check if the edge is the one you added
            nx.draw_networkx_edges(G2, pos, edgelist=[edge], edge_color='green',
                                   width=2.0)  # Set color to red for the added edge
        elif (edge[0], edge[1]) in added_edges or (
        edge[1], edge[0]) in added_edges:  # Check if the edge is the one you added
            nx.draw_networkx_edges(G2, pos, edgelist=[edge], edge_color='red',
                                   width=2.0)  # Set color to red for the added edge
        else:
            nx.draw_networkx_edges(G2, pos, edgelist=[edge], width=1.0)  # Use default color for other edges
    nx.draw_networkx_nodes(G2, pos, alpha=0.3)
    # nx.draw_networkx_labels(G2, pos, labels, font_size=12, font_color='black')
    title = f'Greedy Step {step}'
    plt.title(title, fontsize=100)
    plt.savefig(f'../plots/new_track/greedy/network_step_{step}.png')
    plt.clf()
    with open(rf"../graphs/greedy/new_track_step_{step}.pickle", "wb") as output_file:
        cPickle.dump(G2, output_file)

    return G2, added_edges


if __name__ == "__main__":
    with open(r"../graphs/graph_fixed_attr.pickle", "rb") as input_file:
        G = cPickle.load(input_file)

    x = dict(nx.get_node_attributes(G, 'x'))
    y = dict(nx.get_node_attributes(G, 'y'))
    steps = 10
    top_edges = []
    ef = []
    for step in range(steps):
        node_pairs = [(i, j) for i in G.nodes for j in G.nodes if j > i]
        edges = set(G.edges)
        node_pairs = {(u, v) for u, v in node_pairs if (u, v) not in edges and (v, u) not in edges}
        node_pairs = list(node_pairs)
        # node_pairs = node_pairs[0:200]
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
            i, j, new_efficiency, distance = result
            store[(i, j)] = {
                'new_efficiency': new_efficiency,
                # 'local_efficiency': local_efficiency,
                # 'cluster_efficiency': cluster_efficiency,
                'node1': i,
                'node2': j,
                # 'track_length': track_length,
                'distance': distance
            }

        with open(rf"../files/store/step{step}.pickle", "wb") as output_file:
            cPickle.dump(store, output_file)

        G, added_edges = add_track(store, G, step, top_edges)
        top_edges = top_edges + added_edges
        top_efficiency = nx.global_efficiency(G)
        ef.append(top_efficiency)
        plt.plot(ef)
        plt.xlabel('steps')
        plt.ylabel('Global Efficiency')
        plt.title('Efficiency trough each addition of track')
        plt.savefig(f'../plots/new_track/greedy/curve/curve_step_{step}.png')
        plt.clf()
        with open(f"../log/greedy.txt", "a") as log_file:
            log_file.write(f'{step}\n')
            log_file.write(f'{added_edges}\n')
            log_file.write(f'{top_efficiency}\n')

