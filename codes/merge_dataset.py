import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv('data/NTAD_North_American_Rail_Network_Lines_Passenger_Rail.csv')
print(df.shape)
# df = df.drop_duplicates(subset=['FRFRANODE', 'TOFRANODE'], keep='first').reset_index(drop=True)
df = df[(df['TRKRGHTS1']=='AMTK') | (df['RROWNER1'] == 'AMTK')]
print(df.shape)
# df = df[df['BRANCH']=='MAIN']
# print(df.shape)
df = df.drop_duplicates(subset=['FRFRANODE', 'TOFRANODE'], keep='first').reset_index(drop=True)

print(df.shape)

G = nx.Graph()
edges = [(0, 1), (0, 2), (0, 5), (0, 6), (0, 7), (0, 8), (2, 3), (2, 5), (3, 5), (4, 5)]

for i in range(df.shape[0]):
    G.add_edge(df.loc[i,'FRFRANODE'], df.loc[i,'TOFRANODE'])


len(G.edges)
len(G.nodes)
dir(G)

sorted([i[1] for i in G.degree], reverse=True)
pos2 = nx.spring_layout(G,seed = 42)
plt.figure(figsize=(8, 6))
nx.draw_networkx(G, pos=pos2, with_labels=True)

df['KM'].describe()

df2 = pd.read_csv('data/Amtrak_Routes.csv')
df2.describe()

df[df['TRKRGHTS1']=='AMTK'].count()
