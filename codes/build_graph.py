import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Read the edge dataset
edge_df = pd.read_csv('data/NTAD_North_American_Rail_Network_Lines_Passenger_Rail.csv')
print(edge_df.shape)
# edge_df = edge_df[(edge_df['TRKRGHTS1']=='AMTK') | (edge_df['RROWNER1'] == 'AMTK')] # get the Amtrak tracks we want
print(edge_df.shape)
edge_df = edge_df.drop_duplicates(subset=['FRFRANODE', 'TOFRANODE'], keep='first').reset_index(drop=True)
print(edge_df.shape)
# edge_df.loc[edge_df['OBJECTID']==308556,'TOFRANODE'] = 335062
edge_df.loc[edge_df['OBJECTID']==308829,'FRFRANODE'] = 334415 # Helper
# edge_df.loc[edge_df['OBJECTID']==307006,'TOFRANODE'] = 334351
# edge_df.loc[edge_df['OBJECTID']==298719,'TOFRANODE'] = 494929  # Boston
# edge_df.loc[edge_df['OBJECTID']==79606,'TOFRANODE'] = 494929  # Boston
edge_df.loc[edge_df['OBJECTID']==338923,'TOFRANODE'] = 495049  # Boston
edge_df.loc[edge_df['OBJECTID']==79176,'TOFRANODE'] = 495049  # Boston
edge_df.loc[edge_df['OBJECTID']==79175,'TOFRANODE'] = 495049  # Boston
edge_df.loc[edge_df['OBJECTID']==315490,'TOFRANODE'] = 495049  # Boston
edge_df.loc[edge_df['OBJECTID']==315290,'TOFRANODE'] = 495049  # Boston
edge_df.loc[edge_df['OBJECTID']==333011,'TOFRANODE'] = 448147  # Spartanburg
edge_df.loc[edge_df['OBJECTID']==74688,'FRFRANODE'] = 492597  # Spartanburg


# Read the station dataset
# node_df = pd.read_csv('data/NTAD_North_American_Rail_Network_Nodes.csv')
node_df = pd.read_csv('data/station_data_cleaned.csv') # merged dataset from merge.py
print(node_df.shape)
node_df = node_df[(node_df['PASSNGR'].isin(['A', 'B']))] # Get the Amtrak Station
relabel = {
    335464:335453, # Helper
    352445:352446, # San Antonio
    354415:354413, # San Marcos
    357218:357219, # Taylor
    357262:357256, # McGregor
    357444:357476, # Cleburne
    359372:359436, # Ardmore
    378195:378160, # Texarkana
    395727:395736, # Greenwood
    431778:431782, # Cincinnati
    445007:445193, # Port Huron
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
G2 = copy.deepcopy(G)

# Sort the edge list based on the sum of the node IDs at each end of the edge

while len(set(G2.nodes) - set(stations)) >129: # keep starting over till all edges are merged (there are 126 nodes that don't connected to any station)
    edges_list = copy.deepcopy(G2.edges)
    sorted_edge_list = sorted(edges_list, key=lambda x: sum(x[:2]))

    for edge in sorted_edge_list:
        try:
            if edge[0] not in stations and edge[1] not in stations:
                nx.contracted_edge(G2, (edge[1], edge[0]), copy=False, self_loops=False)
            elif edge[0] in stations and edge[1] not in stations:
                nx.contracted_edge(G2, (edge[0], edge[1]), copy=False, self_loops=False)
            elif edge[0] not in stations and edge[1] in stations:
                nx.contracted_edge(G2, (edge[1], edge[0]), copy=False, self_loops=False)
                # print('success')
        except:
            # print('failed')
            continue
    print(len(set(G2.nodes) - set(stations)), 'redundant remaining')

# simplify the tracks, merge connected inter-tracks
G3 = copy.deepcopy(G)

# Sort the edge list based on the sum of the node IDs at each end of the edge

while len(set(G3.nodes) - set(stations)) >129: # keep starting over till all edges are merged (there are 126 nodes that don't connected to any station)
    edges_list = copy.deepcopy(G3.edges)
    sorted_edge_list = sorted(edges_list, key=lambda x: sum(x[:2]), reverse=True)

    for edge in sorted_edge_list:
        try:
            if edge[0] not in stations and edge[1] not in stations:
                nx.contracted_edge(G3, (edge[1], edge[0]), copy=False, self_loops=False)
            elif edge[0] in stations and edge[1] not in stations:
                nx.contracted_edge(G3, (edge[0], edge[1]), copy=False, self_loops=False)
            elif edge[0] not in stations and edge[1] in stations:
                nx.contracted_edge(G3, (edge[1], edge[0]), copy=False, self_loops=False)
                # print('success')
        except:
            # print('failed')
            continue
    print(len(set(G3.nodes) - set(stations)), 'redundant remaining')

# Create a new graph for the sum
G_sum = nx.Graph()

# Add edges from G1 to G_sum
G_sum.add_edges_from(G2.edges())

# Add edges from G2 to G_sum
G_sum.add_edges_from(G3.edges())

G2 = G_sum

# Remove the 126 nodes that are not connected to any stations
G2.remove_nodes_from(set(set(G2.nodes) - set(stations)))
# G2.add_edge(497523,497525)
# G2.add_edge(497525,497527)
## Set length
distance = nx.shortest_path_length(G, weight='KM')
distance_dict = {}
attr = {}
position = {}
name = {}
for i in G2.nodes:
    t = node_df[node_df['FRANODEID'] == i]
    position[i] = (t['x'].values[0], t['y'].values[0])
    name[i] = t['Station'].values[0]

    attr[i] = {"x": t['x'].values[0], "y": t['y'].values[0], "state":t['State'].values[0], "name":t['Station'].values[0], 'ridership':t['ridership'].values[0]}

    for j in G2.nodes:
        if i==j or (i, j) not in G2.edges:
            continue
        try:
            distance_dict[(i,j)] = nx.shortest_path_length(G, source=i, target=j, weight='KM')
        except:
            distance_dict[(i,j)] = np.inf
nd = {}
for i in distance_dict:
    nd[i] = {'KM': distance_dict[i]}
nx.set_node_attributes(G2, attr)
nx.set_edge_attributes(G2, nd)

set(stations) - set(G2.nodes)
print("=== Original data ===")

print('m', len(G2.edges))
print('n', len(G2.nodes))
print('k', G2.degree)

# for i in stations
sorted(G2.degree)
sorted([i[1] for i in G2.degree])
for i in G2.degree:
    if i[1] == 5:
        print(i)

d = [i[1] for i in G2.degree]
sorted(d,reverse=False)
plt.hist(d)
plt.show()
node_df[node_df['FRANODEID']==321401]
node_df[node_df['PASSNGRSTN']=='Provo Central Station']

len(stations)
sorted([i[1] for i in G.degree], reverse=True)
# pos2 = nx.spring_layout(G, seed=42)
plt.figure(figsize=(30, 18))

# nx.draw_networkx(G2, pos=position, with_labels=False, labels=name)
nx.draw_networkx_edges(G2, pos=position)
plt.title('US Railway Network')
plt.savefig('plots/network.png')



import _pickle as cPickle
with open(r"graphs/graph_fixed.pickle", "wb") as output_file:
    cPickle.dump(G2, output_file)


node_df[node_df['FRANODEID']==465725]
node_df[node_df['FRANODEID']==465762]

node_df[node_df['Station']=='Dallas, Texas']
node_df[node_df['Station']=='Houston, Texas']
import math
math.sqrt((-84.392836 + 81.148882)**2 + (33.799112 - 32.083369)**2)
math.sqrt((-96.808093 + 95.367769)**2 + (32.775818 - 29.767695)**2)

