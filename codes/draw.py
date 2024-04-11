import _pickle as cPickle

import matplotlib.pyplot as plt
import networkx as nx

with open(r"graphs/graph_with_attr.pickle", "rb") as input_file:
    G = cPickle.load(input_file)

nx.get_edge_attributes(G, 'KM')
nx.get_node_attributes(G, 'x')
x = dict(nx.get_node_attributes(G, 'x'))
y = dict(nx.get_node_attributes(G, 'y'))
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
# plt.savefig('network_KM.png')

plt.figure(figsize=(50, 30))
p = nx.draw(G, pos, with_labels=True, node_color=list(nx.get_node_attributes(G, 'betweenness_centrality').values()),
            labels=name)
# plt.colorbar(p)
plt.savefig('plots/network_between.png')

plt.figure(figsize=(50, 30))
p = nx.draw(G, pos, with_labels=True, node_color=list(nx.get_node_attributes(G, 'closeness_centrality').values()),
            labels=name)
# plt.colorbar(p)

plt.savefig('plots/closeness_centrality.png')
