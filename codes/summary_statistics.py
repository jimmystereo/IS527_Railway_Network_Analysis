import copy

import networkx as nx
import _pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt

with open(r"graphs/graph_fixed.pickle", "rb") as input_file:
    G = cPickle.load(input_file)

cc = nx.average_clustering(G)

m = len(G.edges)
n = len(G.nodes)
mean_degree = 2*m/n


print('m: ', m)
print('n: ', n)
print('Mean Degree: ', round(mean_degree,2))
print('Clustering Coefficient: ', round(cc,2))



G = G.to_undirected()
## eigenvector_centrality
centrality = nx.eigenvector_centrality(G)
nx.set_node_attributes(G, {i:{'eigenvector_centrality': centrality[i]} for i in centrality})
k = []
for i in G.degree:
    k.append(i[1]/m)
fig, ax = plt.subplots()
c = [centrality[i] for i in centrality.keys()]
ax.scatter(k, c)
n = nx.get_node_attributes(G,'name')
n = [n[i] for i in n]
for i, txt in enumerate(n):
    ax.annotate(txt, (k[i], c[i]))
plt.xlabel('normalized degree, k/m')
plt.ylabel('eigenvector centrality, x')
plt.title('Eigenvector Centrality')
# plt.savefig('network_KM.png')

## PageRank
pr = nx.pagerank(G, alpha=0.85)
nx.set_node_attributes(G, {i:{'PageRank': pr[i]} for i in pr})

fig, ax = plt.subplots()
pr = [pr[i] for i in pr.keys()]

ax.scatter(k, pr)
# n[3:-1] = ''
for i, txt in enumerate(n):
    ax.annotate(txt, (k[i], pr[i]))
plt.xlabel('normalized degree, k/m')
plt.ylabel('PageRank, y')
plt.title('PageRank of the network')


## closeness centrality
close = nx.closeness_centrality(G,distance = 'KM')
nx.set_node_attributes(G, {i:{'closeness_centrality': close[i]} for i in close})

fig, ax = plt.subplots()
close = [close[i] for i in close.keys()]

ax.scatter(k, close)
# n[3:-1] = ''
for i, txt in enumerate(n):
    ax.annotate(txt, (k[i], close[i]))
plt.xlabel('normalized degree, k/m')
plt.ylabel('Closeness centrality, c')
plt.title('Closeness centrality of the network')


## Harmonic centrality
h = nx.harmonic_centrality(G,distance = 'KM')
nx.set_node_attributes(G, {i:{'harmonic_centrality': h[i]} for i in h})

fig, ax = plt.subplots()
h = [h[i] for i in h.keys()]

ax.scatter(k, h)
# n[3:-1] = ''
for i, txt in enumerate(n):
    ax.annotate(txt, (k[i], h[i]))
plt.xlabel('normalized degree, k/m')
plt.ylabel('Harmonic centrality, c')
plt.title('Harmonic centrality of the network')

## Betweenness centrality
b = nx.betweenness_centrality(G,weight = 'KM')
nx.set_node_attributes(G, {i:{'betweenness_centrality': b[i]} for i in b})

fig, ax = plt.subplots()
b = [b[i] for i in b.keys()]

ax.scatter(k, b)
# n[3:-1] = ''
for i, txt in enumerate(n):
    ax.annotate(txt, (k[i], b[i]))
plt.xlabel('normalized degree, k/m')
plt.ylabel('Betweenness centrality, c')
plt.title('Betweenness centrality of the network')


import _pickle as cPickle
with open(r"graphs/graph_fixed_attr.pickle", "wb") as output_file:
    cPickle.dump(G, output_file)