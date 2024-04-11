import pandas as pd
import _pickle as cPickle
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import networkx as nx


df2 = pd.read_csv('data/ridership_table.csv', encoding='utf-8')
df2 = df2.iloc[:,[0,1,3]]
df2 = df2.dropna()
# Read the station dataset
node_df = pd.read_csv('data/NTAD_North_American_Rail_Network_Nodes.csv')
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

node_df = node_df.rename(columns={'STATE': 'Abbreviation'})

state = pd.read_csv('data/states.csv')
node_df = pd.merge(node_df, state, on = 'Abbreviation',how='left')
node_df['Station'] = node_df['PASSNGRSTN']+', '+node_df['State']
#
list1 = node_df['Station'].tolist()
list2 = df2['Station'].tolist()
mat1 = []
mat2 = []
# taking the threshold as 80
threshold = 90
for i in list1:
    mat1.append(process.extract(i, list2, limit=1))
node_df['matches'] = mat1

node_df['score'] = node_df['matches'].apply(lambda x: x[0][1])
node_df['matches'] = node_df['matches'].apply(lambda x: x[0][0])
node_df['Station'].value_counts()
# node_df = node_df[node_df['Station']==86]
# len(list1)
# node_df = pd.merge(node_df, df2, on = 'Station',how='left')
node_df.shape
node_df.to_csv('data/merged_dataset.csv', index=False)


df = pd.read_csv('data/merged_dataset_cleaned.csv')
df = df[df['fixed']!='drop']
df = df.drop_duplicates(subset = 'fixed')
df['Station'] = df['fixed']
df = df.drop(columns = ['matches','score'])
df3 = df.merge(df2.drop(columns = ['State']), on = 'Station', how = 'left')
df3 = df3.drop(columns = ['fixed'])
df3 = df3.rename(columns = {'2022':'ridership'})
df3 = df3.dropna(subset = ['ridership'])
df3['ridership'] = df3['ridership'].apply(lambda x: int(x.replace(',', '')))
df3.to_csv('data/station_data_cleaned.csv', index=False)
df['fixed'].value_counts()
df3.sort_values(by = 'ridership', ascending =False)