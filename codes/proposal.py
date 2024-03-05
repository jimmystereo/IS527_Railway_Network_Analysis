import pandas as pd
df = pd.read_csv('data/NTAD_North_American_Rail_Network_Lines_Passenger_Rail_-1818839717552070515.csv')
#%%
df.groupby(['FRFRANODE', 'TOFRANODE'])['OBJECTID'].count().reset_index().sort_values('OBJECTID',ascending=False)
#%%
df.describe()
#%%
df[~df.isna()].count()
#%%
df.head()

#%%
df2 = pd.read_csv('data/table.csv', encoding='utf-8')

df2
#%%
df4 = pd.read_csv('data/Amtrak_Stations.csv', encoding='utf')
df
#%%
#
# dfr = df
# dfr['FRFRANODE'] = dfr['TOFRANODE']
# dfr['TOFRANODE'] = df['FRFRANODE']
df.head()
suc = 0
fail = 0
for i in range(df.shape[0]):
    node1 = df.iloc[i,2]
    node2 = df.iloc[i,3]
    if df[df['FRFRANODE']==node2][df['TOFRANODE']==node1].shape[0] != 0:
        suc += 1
    else:
        fail+=1
print(suc, fail)
print(df.shape)
import pandas as pd
df = pd.read_csv('data/NTAD_North_American_Rail_Network_Lines_Passenger_Rail_-1818839717552070515.csv')
#%%
df.groupby(['FRFRANODE', 'TOFRANODE'])['OBJECTID'].count().reset_index().sort_values('OBJECTID',ascending=False)
#%%
df.describe()
#%%
df[~df.isna()].count()
#%%
df.head()

#%%
df2 = pd.read_csv('data/table.csv', encoding='utf-8')

df2
#%%
df4 = pd.read_csv('data/Amtrak_Stations.csv', encoding='utf')
df
#%%
#
# dfr = df
# dfr['FRFRANODE'] = dfr['TOFRANODE']
# dfr['TOFRANODE'] = df['FRFRANODE']
df.head()
suc = 0
fail = 0
for i in range(df.shape[0]):
    node1 = df.iloc[i,2]
    node2 = df.iloc[i,3]
    if df[df['FRFRANODE']==node2][df['TOFRANODE']==node1].shape[0] != 0:
        suc += 1
    else:
        fail+=1
print(suc, fail)
print(df.shape)

