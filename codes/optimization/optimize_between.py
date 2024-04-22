import _pickle as cPickle
import networkx as nx
from multiprocessing import Pool

def calculate_metrics_batch(args):
    node_pairs, G, x, y, progress_callback = args
    results = []
    total_pairs = len(node_pairs)
    for idx, node_pair in enumerate(node_pairs, 1):
        i, j = node_pair
        G.add_edge(i, j)
        distance = ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) ** 0.5
        attrs = {(i, j): {"KM": distance}}
        nx.set_edge_attributes(G, attrs)
        new_efficiency = nx.global_efficiency(G)
        try:
            edge_betweenness = nx.edge_betweenness_centrality(G, weight='KM')[(i, j)]
        except:
            edge_betweenness = nx.edge_betweenness_centrality(G, weight='KM')[(j, i)]
        G.remove_edge(i, j)
        results.append((i, j, new_efficiency, edge_betweenness, distance))
        progress_callback(idx, total_pairs)
    return results

def print_progress(progress, total):
    percent_complete = (progress / total) * 100
    print(f"Progress: {progress}/{total} ({percent_complete:.2f}%)")

if __name__ == "__main__":
    with open(r"../graphs/graph_fixed_attr.pickle", "rb") as input_file:
        G = nx.Graph(cPickle.load(input_file))
    x = dict(nx.get_node_attributes(G, 'x'))
    y = dict(nx.get_node_attributes(G, 'y'))

    node_pairs = [(i, j) for i in G.nodes for j in G.nodes if j > i]
    edges = set(G.edges)
    node_pairs = [(u, v) for u, v in node_pairs if (u, v) not in edges and (v, u) not in edges]

    total_pairs = len(node_pairs)
    num_processes = 4  # Adjust according to the number of CPU cores available
    batch_size = total_pairs // num_processes

    args_list = [(node_pairs[i:i+batch_size], G.copy(), x, y, print_progress) for i in range(0, total_pairs, batch_size)]

    with Pool(num_processes) as pool:
        results = []
        for result_batch in pool.imap_unordered(calculate_metrics_batch, args_list):
            results.extend(result_batch)

    store = {(i, j): {'new_efficiency': new_efficiency, 'edge_betweeness': edge_betweenness, 'distance': distance}
             for i, j, new_efficiency, edge_betweenness, distance in results}

    with open(r"../files/proposed_between.pickle", "wb") as output_file:
        cPickle.dump(store, output_file)
