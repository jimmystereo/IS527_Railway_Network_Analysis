import networkx as nx
import _pickle as cPickle


with open(r"graph.pickle", "rb") as input_file:
    G = cPickle.load(input_file)

cc = nx.average_clustering(G)

m = len(G.edges)
n = len(G.nodes)
mean_degree = 2*m/n


print('m: ', m)
print('n: ', n)
print('Mean Degree: ', mean_degree)
print('Clustering Coefficient: ', cc)

# nx.average_shortest_path_length(G)
# nx.draw_networkx(G)