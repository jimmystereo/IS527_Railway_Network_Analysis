import _pickle as cPickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
import pandas as pd
import copy
import statistics
import math

with open(r"../graphs/graph_fixed_attr.pickle", "rb") as input_file:
    G = cPickle.load(input_file).to_undirected()

with open(r"../files/proposed.pickle", "rb") as input_file:
    store = cPickle.load(input_file)

def dist_between(node1, node2):
    return math.sqrt((x[node1]-x[node2])**2+(y[node1]-y[node2])**2)



def standardize(x, mean, std):
    return (x - mean) / std

def normalize(x, max, min):
    min = 0.99*min
    return (x - min) / (max- min)
normalized_store = copy.deepcopy(store)
for key in ['new_efficiency','local_efficiency','track_length', 'distance', 'cluster_efficiency']:
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
alpha = 0.5
degrees = dict(nx.degree(G))
for i,j in store.keys():
    # delta_efficiency = efficiency-store[(i,j)]["new_efficiency"]
    track_length = store[(i,j)]["track_length"]
    distance = store[(i,j)]["distance"]
    cluster_efficiency = store[(i,j)]["cluster_efficiency"]
    # coefficient = store[(i,j)]["cluster_efficiency"]**alpha*(ridership[i]/degrees[i]+ridership[j]/degrees[j])*(track_length/distance)**beta
    # coefficient = store[(i,j)]["cluster_efficiency"]**alpha*(track_length/distance)**beta
    # coefficient = store[(i,j)]["local_efficiency"]*avg_neighbor_ridership[i]*avg_neighbor_ridership[j]*(track_length/distance)**beta
    # coefficient = store[(i,j)]["local_efficiency"] - alpha*(ridership[i]*ridership[j])
    coefficient = (store[(i,j)]["new_efficiency"]**2 - (ridership[i]*ridership[j]))/distance
    # coefficient = (store[(i,j)]["new_efficiency"]*store[(i,j)]["cluster_efficiency"] - alpha*(ridership[i]*ridership[j]))
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

tmp = df.sort_values(by = 'coefficient', ascending = False).reset_index(drop = True)
tmp['coefficient']/tmp['distance']
G2 = copy.deepcopy(G)
top_edges = []
i = -1
c = 1
highbar = df['distance'].quantile(.05)
# highbar = normalized_store(3.669742715254712, max([item['distance'] for item in store.values()]),min([item['distance'] for item in store.values()]))
low = df['distance'].quantile(.01)
# df = df[df['distance'] <=  highbar]
# df = df[df['distance'] >=  low]
# df = df[df['track_length'] < df['track_length'].quantile(.2)]
top_df = df.sort_values(by = 'coefficient', ascending = False).reset_index(drop = True)

while c <=50:
    i+=1
    tops = top_df.loc[i,['node1','node2']]
    node1 = int(tops["node1"])
    node2 = int(tops["node2"])
    if (node1, node2) in G2.edges:
        print(node1, node2)
        continue
    G2.add_edge(node1,node2)
    top_edges.append((node1,node2))
    c+=1


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
        print(edge)
        nx.draw_networkx_edges(G2, pos, edgelist=[edge], edge_color='red', width=2.0)  # Set color to red for the added edge
    else:
        nx.draw_networkx_edges(G2, pos, edgelist=[edge], width=1.0)  # Use default color for other edges
nx.draw_networkx_nodes(G2, pos, alpha=0.3)
# nx.draw_networkx_labels(G2, pos, labels, font_size=12, font_color='black')
title = 'coefficient = (store[(i,j)]["global"]*store[(i,j)]["cluster_efficiency"] - alpha*(ridership[i]*ridership[j]*distance))'
plt.title(title, fontsize = 80)
plt.savefig('../plots/new_track/new/network_opt_39.png')
plt.show()

import _pickle as cPickle
with open(r"../graphs/new_track.pickle", "wb") as output_file:
    cPickle.dump(G2, output_file)
# node_attributes = ("ridership",)
# edge_attributes = ("KM",)
# summary_graph = nx.snap_aggregation(
#     G, node_attributes=node_attributes, edge_attributes=edge_attributes)
#
# plt.figure(figsize=(75, 45))
# pos2 = nx.spring_layout(summary_graph,seed = 42)
#
# for edge in summary_graph.edges():
#     if set(edge) in top_edges:  # Check if the edge is the one you added
#         nx.draw_networkx_edges(summary_graph, pos2, edgelist=[edge], edge_color='red', width=2.0)  # Set color to red for the added edge
#     else:
#         nx.draw_networkx_edges(summary_graph, pos2, edgelist=[edge], width=1.0)  # Use default color for other edges
#
# nx.draw_networkx_nodes(summary_graph, pos2, alpha=0.7)
# nx.draw_networkx_labels(summary_graph, pos2, labels, font_size=12, font_color='black')
# plt.title('new_efficiency*(track_length/distance)_distance<5%')
# plt.savefig('plots/summary_graph.png')
# plt.show()
#
# for edge in G2.edges():
#     if edge in top_edges:  # Check if the edge is the one you added
#         print(edge)
# ( 497529,319565) in G2.edges
# (319565, 321401) in top_edges
#
# for edge in G2.edges():
#     if (edge[0], edge[1]) in top_edges or (edge[1], edge[0]) in top_edges:  # Check if the edge is the one you added
#         print(edge)