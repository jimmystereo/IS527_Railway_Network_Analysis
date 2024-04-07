import pandas as pd
import networkx as nx
import copy
import matplotlib.pyplot as plt
import numpy as np
# Read the edge dataset
edge_df = pd.read_csv('data/NTAD_North_American_Rail_Network_Lines_Passenger_Rail.csv')
print(edge_df.shape)
# edge_df = edge_df[(edge_df['TRKRGHTS1']=='AMTK') | (edge_df['RROWNER1'] == 'AMTK')] # get the Amtrak tracks we want
print(edge_df.shape)
edge_df = edge_df.drop_duplicates(subset=['FRFRANODE', 'TOFRANODE'], keep='first').reset_index(drop=True)
print(edge_df.shape)

# Read the station dataset
node_df = pd.read_csv('data/NTAD_North_American_Rail_Network_Nodes.csv')
print(node_df.shape)
node_df = node_df[(node_df['PASSNGR'].isin(['A', 'B']))] # Get the Amtrak Station
relabel = {
    335464:335453, #Helper
    352445:352446, # San Antonio
    354415:354413, # San Marcos
    357218:357219, # Taylor
    357262:357256, # McGregor
    357444:357476, # Cleburne
    359372:359436, # Ardmore
    378195:378160, # Texarkana
    395727:395736, # Greenwood
    431778:431782, # Cincinnati
    445007: 445193, # Port Huron
    466120:465762, # Latrobe
    467787:	467652 # Johnstown
}
def relabel_node(x):
    if x in relabel:
        return relabel[x]
    return x
node_df['FRANODEID'] =node_df['FRANODEID'].apply(relabel_node)
print(node_df.shape)

stations = set(node_df['FRANODEID'].values) # The station list
stations.remove(480266)


# Create a graph we all the tracks
G = nx.Graph()
for i in range(edge_df.shape[0]):
    G.add_edge(edge_df.loc[i,'FRFRANODE'], edge_df.loc[i,'TOFRANODE'],KM = edge_df.loc[i,'KM'])
G = G.to_undirected()
print("=== Original data ===")
print('m', len(G.edges))
print('n', len(G.nodes))


# simplify the tracks, merge connected inter-tracks
G2 = G.copy()
while len(set(G2.nodes) - set(stations)) >126: # keep starting over till all edges are merged (there are 126 nodes that don't connected to any station)
    edges_list = copy.deepcopy(G2.edges)
    for edge in edges_list:
        try:
            if edge[0] not in stations and edge[1] not in stations:
                nx.contracted_edge(G2, (edge[1], edge[0]), copy=False, self_loops=False)
            elif edge[0] in stations and edge[1] not in stations:
                nx.contracted_edge(G2, (edge[0], edge[1]), copy=False, self_loops=False)
            elif edge[0] not in stations and edge[1] in stations:
                nx.contracted_edge(G2, (edge[1], edge[0]), copy=False, self_loops=False)
        except:
            continue
    print(len(set(G2.nodes) - set(stations)), 'redundant remaining')

# Remove the 126 nodes that are not connected to any stations
G2.remove_nodes_from(set(set(G2.nodes) - set(stations)))

## Set length
distance = nx.shortest_path_length(G, weight='KM')
distance_dict = {}
attr = {}
position = {}
name = {}
for i in G2.nodes:
    t = node_df[node_df['FRANODEID'] == i]
    position[i] = (t['x'].values[0], t['y'].values[0])
    name[i] = t['PASSNGRSTN'].values[0]

    attr[i] = {"x": t['x'].values[0], "y": t['y'].values[0], "state":t['STATE'].values[0], "name":t['PASSNGRSTN'].values[0]}

    # for j in G2.nodes:
    #     if i==j:
    #         continue
    #     try:
    #         distance_dict[(i,j)] = nx.shortest_path_length(G, source=i, target=j, weight='KM')
    #     except:
    #         distance_dict[(i,j)] = np.inf
nx.set_node_attributes(G2, attr)
nx.get_node_attributes(G2,'x')

# nx.set_edge_attributes(G2, distance_dict, "KM")
# nx.get_edge_attributes(G2,'KM')

set(stations) - set(G2.nodes)
print("=== Original data ===")

print('m', len(G2.edges))
print('n', len(G2.nodes))
print('k', G2.degree)

# for i in stations
sorted(G2.degree)
sorted([i[1] for i in G2.degree])
for i in G2.degree:
    if i[1] == 0:
        print(i)

d = [i[1] for i in G2.degree]
sorted(d,reverse=False)
plt.hist(d)
node_df[node_df['FRANODEID']==480266]
len(stations)
sorted([i[1] for i in G.degree], reverse=True)
# pos2 = nx.spring_layout(G, seed=42)
plt.figure(figsize=(30, 18))

nx.draw_networkx(G2, pos=position, with_labels=True, labels=name)
plt.title('US Railway Network')
plt.savefig('network.png')

node_df['PASSNGR'].value_counts()



import _pickle as cPickle
with open(r"graph.pickle", "wb") as output_file:
    cPickle.dump(G2, output_file)





