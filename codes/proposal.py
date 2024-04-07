import pandas as pd
df = pd.read_csv('data/NTAD_North_American_Rail_Network_Lines_Passenger_Rail.csv', dtype={
    'STCNTYFIPS': 'string'
})
df
#%%
df.groupby(['FRFRANODE', 'TOFRANODE'])['OBJECTID'].count().reset_index().sort_values('OBJECTID',ascending=False)
#%%
df.describe()
#%%
df[~df.isna()].count()
#%%
df.head()

#%%
df2 = pd.read_csv('data/ridership_table.csv', encoding='utf-8')

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
df = pd.read_csv('data/NTAD_North_American_Rail_Network_Lines_Passenger_Rail.csv')
#%%
df.groupby(['FRFRANODE', 'TOFRANODE'])['OBJECTID'].count().reset_index().sort_values('OBJECTID',ascending=False)
#%%
df.describe()
#%%
df[~df.isna()].count()
#%%
df.head()

#%%
df2 = pd.read_csv('data/ridership_table.csv', encoding='utf-8')

df2
#%%
df4 = pd.read_csv('data/Amtrak_Stations.csv', encoding='utf')
df4
df4[df4['stationnam']=="REESE, MS"]
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

df5 = pd.read_csv('data/Rail_Equipment_Accident_Incident_Data__Form_54__20240306.csv')
df5
df5['Station']
df
t1[t1['STCNTYFIPS'].isnull()]
df[df['STCNTYFIPS'] == '03004']
t1.shape
df6 = pd.read_csv('data/state_and_county_fips_master.csv',dtype={'fips':'string'})
t1= df[['STCNTYFIPS', 'OBJECTID' ]]
t1['STCNTYFIPS'] = t1['STCNTYFIPS'].apply(lambda x : int(x))
t1.iloc[:, t1['STCNTYFIPS'].isnull()]
t2 = df6
df7 = pd.merge(t1, t2, left_on='STCNTYFIPS', right_on='fips', how = 'inner')
df7
"REESE, MS"

