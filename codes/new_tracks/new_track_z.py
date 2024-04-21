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
    G = cPickle.load(input_file)

with open(r"../files/proposed.pickle", "rb") as input_file:
    store = cPickle.load(input_file)

def dist_between(node1, node2):
    return ((x[node1]-x[node2])**2+(y[node1]-y[node2])**2)**(0.5)



def standardize(x, mean, std):
    return (x - mean) / std

def normalize(x, max, min):
    return (x - min) / (max- min)
standardized_store = copy.deepcopy(store)
for key in ['new_efficiency','track_length', 'distance']:
    mean = statistics.mean([item[key] for item in store.values()])
    std = statistics.stdev([item[key] for item in store.values()])
    for i in standardized_store.keys():
        standardized_store[i][key] = standardize(standardized_store[i][key], mean, std)
store = standardized_store

x = dict(nx.get_node_attributes(G, 'x'))
y = dict(nx.get_node_attributes(G, 'y'))
ridership = dict(nx.get_node_attributes(G, 'ridership'))
mean = statistics.mean(ridership.values())
std = statistics.stdev(ridership.values())
for i in G.nodes:
    ridership[i] = standardize(ridership[i], mean, std)
efficiency = nx.global_efficiency(G)
beta = 3
alpha = 2
for i,j in store.keys():
    # delta_efficiency = efficiency-store[(i,j)]["new_efficiency"]
    track_length = store[(i,j)]["track_length"]
    distance = store[(i,j)]["distance"]
    coefficient = store[(i,j)]["new_efficiency"]**alpha*math.sqrt(ridership[i]*ridership[j])*(track_length/distance)
    # coefficient = store[(i,j)]["new_efficiency"]*(track_length/distance)
    # coefficient = (ridership[i]+ridership[j])*(track_length/distance*100)
    # coefficient = (track_length/distance*100)
    store[(i,j)]['coefficient'] = coefficient


# Print or use sorted_store as needed
df = pd.DataFrame(store).T
df = df.reset_index(drop=True)
G2 = copy.deepcopy(G)
top_edges = []
i = -1
c = 1
df = df[df['distance'] <  df['distance'].quantile(.05)]
top_df = df.sort_values(by = 'coefficient', ascending = False).reset_index(drop = True)

while c <=5:
    i+=1
    tops = top_df.loc[i,['node1','node2']]
    node1 = int(tops["node1"])
    node2 = int(tops["node2"])
    if (node1, node2) in G2.edges:
        print(1)
        continue
    G2.add_edge(node1,node2)
    top_edges.append((node1,node2))
    c+=1


# Define positions, edge labels, and node labels
pos = {}
KM = nx.get_edge_attributes(G2, 'KM')
name = nx.get_node_attributes(G2, 'name')
KM = {i: round(KM[i], 2) for i in KM}
for i in x:
    pos[i] = (x[i], y[i])
labels = {node: name[node] for node in G2.nodes}

# Draw the network with different edge colors
plt.figure(figsize=(75, 45))
for edge in G2.edges():
    if (edge[0], edge[1]) in top_edges or (edge[1], edge[0]) in top_edges:  # Check if the edge is the one you added
        nx.draw_networkx_edges(G2, pos, edgelist=[edge], edge_color='red', width=2.0)  # Set color to red for the added edge
    else:
        nx.draw_networkx_edges(G2, pos, edgelist=[edge], width=1.0)  # Use default color for other edges

nx.draw_networkx_nodes(G2, pos, alpha=0.7)
nx.draw_networkx_labels(G2, pos, labels, font_size=12, font_color='black')
plt.title('"new_efficiency**2*(ridership[i]*ridership[j])**0.5*(track_length/distance)_distance<5%')
plt.savefig('plots/new_track/std/network_opt_14.png')
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