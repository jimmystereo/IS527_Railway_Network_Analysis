import _pickle as cPickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time

with open(r"../graphs/graph_fixed_attr.pickle", "rb") as input_file:
    G = cPickle.load(input_file)
len(G.nodes)
nx.global_efficiency(G)
G.nodes

G.add_edge(404059, 389582)

def dist_between(node1, node2):
    return ((x[node1]-x[node2])**2+(y[node1]-y[node2])**2)**(0.5)

best_combo = {
    "delta_efficiency": 0,
    "coefficient":0,
    "track_length": None,
    "node1": None,
    "node2": None,
}
x = dict(nx.get_node_attributes(G, 'x'))
y = dict(nx.get_node_attributes(G, 'y'))
ridership = dict(nx.get_node_attributes(G, 'ridership'))
efficiency = nx.global_efficiency(G)

for i in G.nodes:
    for j in G.nodes:
        if j <= i:
            continue
        if i!=j and (i, j) not in G.edges:
            # start_time = time.time()

            G.add_edge(i,j)
            new_efficiency = nx.global_efficiency(G)
            delta_efficiency = new_efficiency - efficiency
            track_length =  nx.shortest_path_length(G, source=i, target=j, weight='KM')
            coefficient = delta_efficiency*ridership[i]*ridership[j]*track_length/dist_between(i, j)
            G.remove_edge(i,j)
            # efficiency_time = time.time() - start_time
            # print(efficiency_time)

            if coefficient > best_combo['coefficient']:
                best_combo['delta_coefficieny'] = delta_efficiency
                best_combo['node1'] = i
                best_combo['node2'] = j
                best_combo['track_length'] = track_length
                best_combo['coefficient'] = coefficient




list(G.nodes)[0]
dist_between(315438,  313703)
pos = {}
KM = nx.get_edge_attributes(G, 'KM')
name = nx.get_node_attributes(G, 'name')
KM = {i: round(KM[i], 2) for i in KM}
for i in x:
    pos[i] = (x[i], y[i])

plt.figure(figsize=(75, 45))
nx.draw_networkx(G, pos=pos, with_labels=True, labels=name)
nx.draw_networkx_edge_labels(G, pos, edge_labels=KM, font_color='red')
plt.title('US Railway Network')
# plt.savefig('plots/network_KM_opt.png')
