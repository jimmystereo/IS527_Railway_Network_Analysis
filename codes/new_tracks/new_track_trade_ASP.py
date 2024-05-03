import _pickle as cPickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
import pandas as pd
import copy
import statistics
import math
from sklearn import metrics

with open(r"../graphs/graph_fixed_attr.pickle", "rb") as input_file:
    G = cPickle.load(input_file).to_undirected()

def network_summary(graph):
    summary = {}

    # Number of nodes
    summary['num_nodes'] = graph.number_of_nodes()

    # Number of edges
    summary['num_edges'] = graph.number_of_edges()

    # Average degree
    degree_dict = dict(graph.degree())
    summary['avg_degree'] = sum(degree_dict.values()) / summary['num_nodes']

    # Density
    summary['density'] = nx.density(graph)

    # Diameter
    summary['diameter'] = nx.diameter(graph)

    # Average shortest path length
    summary['avg_shortest_path_length'] = nx.average_shortest_path_length(graph)

    # Clustering coefficient
    summary['clustering_coefficient'] = nx.average_clustering(graph)
    return summary
# network_summary(G)
with open(r"../files/proposed_ASP.pickle", "rb") as input_file:
    store = cPickle.load(input_file)

def robustness(G2):
    result = {}
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
    result['Lcc'] =  metrics.auc(fractions, lccs)


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
            # lcc_size = calculate_weighted_global_efficiency(G_copy)

            fractions.append(fraction)

            lccs.append(lcc_size)
        except:
            continue
    result['Efficiency'] =  metrics.auc(fractions, lccs)
    return result
import networkx as nx

def calculate_weighted_global_efficiency(G):
    """
    Calculate the global efficiency of a graph considering edge weights.

    Parameters:
        G (networkx.Graph): The graph.

    Returns:
        float: The global efficiency.
    """
    # Calculate the shortest path lengths considering edge weights
    shortest_paths = nx.all_pairs_dijkstra_path_length(G, weight='KM')

    # Initialize variables for calculating global efficiency
    total_efficiency = 0
    total_pairs = 0

    # Iterate over pairs of nodes and their shortest path lengths
    for source, target_lengths in shortest_paths:
        for target, length in target_lengths.items():
            if length == 0:
                continue  # Skip nodes that are not connected
            total_efficiency += 1 / length
            total_pairs += 1

    # Calculate the global efficiency
    global_efficiency = total_efficiency / total_pairs

    return global_efficiency

# Example usage:
# Assuming G is your graph
# global_efficiency = calculate_weighted_global_efficiency(G)

def dist_between(node1, node2):
    return math.sqrt((x[node1]-x[node2])**2+(y[node1]-y[node2])**2)



def standardize(x, mean, std):
    return (x - mean) / std

def normalize(x, max, min):
    min = 0.99*min
    return (x - min) / (max- min)
normalized_store = copy.deepcopy(store)
for key in ['ASP', 'distance']:
    max_num = max([item[key] for item in store.values()])
    min_num = min([item[key] for item in store.values()])
    for i in normalized_store.keys():
        normalized_store[i][key] = normalize(normalized_store[i][key], max_num, min_num)
store = normalized_store



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
        ridership_n += ridership[n2]/len(neighbors[n])
    avg_neighbor_ridership[n] = ridership_n
efficiency = nx.global_efficiency(G)
beta =1
alpha = 0
degrees = dict(nx.degree(G))
for i,j in store.keys():
    # delta_efficiency = efficiency-store[(i,j)]["new_efficiency"]
    # track_length = store[(i,j)]["track_length"]
    distance = store[(i,j)]["distance"]
    # cluster_efficiency = store[(i,j)]["cluster_efficiency"]
    # coefficient = store[(i,j)]["cluster_efficiency"]**alpha*(ridership[i]/degrees[i]+ridership[j]/degrees[j])*(track_length/distance)**beta
    # coefficient = store[(i,j)]["cluster_efficiency"]**alpha*(track_length/distance)**beta
    # coefficient = store[(i,j)]["local_efficiency"]*avg_neighbor_ridership[i]*avg_neighbor_ridership[j]*(track_length/distance)**beta
    # coefficient = store[(i,j)]["local_efficiency"] - alpha*(ridership[i]*ridership[j])
    # coefficient = (store[(i,j)]["track_length"] - alpha*(ridership[i]+ridership[j]))/distance
    coefficient = (1/store[(i,j)]["ASP"])**2- alpha*(ridership[i]*ridership[j])
    # coefficient = (store[(i,j)]["cluster_efficiency"]*store[(i,j)]["cluster_efficiency"] - alpha*(ridership[i]*ridership[j]))/distance
    # coefficient = store[(i,j)]["local_efficiency"]*store[(i,j)]["local_efficiency"] - alpha*(ridership[i]*ridership[j])
    # coefficient = store[(i,j)]["new_efficiency"]*(track_length/distance)**beta
    # coefficient = store[(i,j)]["local_efficiency"]*(track_length/distance)**beta
    # coefficient = (ridership[i]+ridership[j])*(track_length/distance*100)
    # coefficient = (track_length/distance*100)
    store[(i,j)]['coefficient'] = coefficient


# Print or use sorted_store as needed
df = pd.DataFrame(store).T
df = df.reset_index(drop=True)

iterations = {
    "thresholds": [],
    "efficiency_scores": [],
    "lcc_scores": [],
    "edges": []
}


highbar = df['distance'].max()
num_tracks = 1
while True:
    print('Threshold', highbar)
    iterations['thresholds'].append(highbar)
    i = -1
    G2 = copy.deepcopy(G)
    top_edges = []
    c = 1
    # highbar = normalized_store(3.669742715254712, max([item['distance'] for item in store.values()]),min([item['distance'] for item in store.values()]))
    # low = df['distance'].quantile(.01)
    df = df[df['distance'] <  highbar]
    # df = df[df['distance'] >=  low]
    # df = df[df['track_length'] < df['track_length'].quantile(.2)]
    top_df = df.sort_values(by = 'coefficient', ascending = False).reset_index(drop = True)
    diss = []
    while c <=num_tracks and i < top_df.shape[0]-1:
        i+=1
        tops = top_df.loc[i,['node1','node2']]
        node1 = int(tops["node1"])
        node2 = int(tops["node2"])
        if (node1, node2) in G2.edges:
            continue
        diss.append(top_df['distance'].iloc[i])
        G2.add_edge(node1,node2)
        top_edges.append((node1,node2))
        c+=1
    if len(diss) < num_tracks:
        iterations['thresholds'] = iterations['thresholds'][0:-1]
        break
    result = robustness(G2)
    iterations["efficiency_scores"].append(result['Efficiency'])
    iterations["lcc_scores"].append(result['Lcc'])
    iterations['edges'].append(top_edges)

    highbar = max(diss)

e = robustness(G)
es = [e['Efficiency'] for _ in range(len(iterations['thresholds']))]
plt.plot(iterations['thresholds'], es,  label = 'Original')
plt.plot(iterations['thresholds'], iterations["efficiency_scores"], label = 'Efficiency')
plt.legend(loc = 'best')
plt.show()

es = [e['Lcc'] for _ in range(len(iterations['thresholds']))]
plt.plot(iterations['thresholds'], es,  label = 'Original')
plt.plot(iterations['thresholds'], iterations["lcc_scores"], label = 'Lcc')
plt.legend(loc = 'best')
plt.show()

max_index = iterations["efficiency_scores"].index(max(iterations["efficiency_scores"]))
top_edges = iterations['edges'][max_index]
iter_df = pd.DataFrame(iterations)

top_edges = iterations['edges'][iter_df[iter_df['thresholds'] < 0.2]['efficiency_scores'].idxmax()]
G2 = copy.deepcopy(G)
G2.add_edges_from(top_edges)
if True:
    # Define positions, edge labels, and node labels
    pos = {}
    KM = nx.get_edge_attributes(G2, 'KM')
    name = nx.get_node_attributes(G2, 'name')
    KM = {i: round(KM[i], 2) for i in KM}
    for i in G2.nodes:
        pos[i] = (x[i], y[i])
    labels = {node: name[node] for node in G2.nodes}

    # Draw the network with different edge colors
    plt.figure(figsize=(150, 90))
    for edge in G2.edges():
        if (edge[0], edge[1]) in top_edges or (edge[1], edge[0]) in top_edges:  # Check if the edge is the one you added
            nx.draw_networkx_edges(G2, pos, edgelist=[edge], edge_color='red', width=2.0)  # Set color to red for the added edge
        else:
            nx.draw_networkx_edges(G2, pos, edgelist=[edge], width=1.0)  # Use default color for other edges
    nx.draw_networkx_nodes(G2, pos, alpha=0.3)
    # nx.draw_networkx_labels(G2, pos, labels, font_size=12, font_color='black')
    title = 'coefficient = (store[(i,j)]["global"]*store[(i,j)]["cluster_efficiency"] - alpha*(ridership[i]*ridership[j]*distance))'
    plt.title(title, fontsize = 80)
    plt.savefig('../plots/new_track/new/network_opt_43.png')
    plt.show()

    # import _pickle as cPickle
with open(r"../files/iterations/ASP_1tracks_rider_0.pickle", "wb") as output_file:
    cPickle.dump(iterations, output_file)

