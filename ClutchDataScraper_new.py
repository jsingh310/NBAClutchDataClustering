#read libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from adjustText import adjust_text
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

#url + headers
#url = 'http://stats.inpredictable.com/nba/ssnPlayerSplit.php?season=2019&team=ALL&pos=ALL&po=0&frdt=2019-10-22&todt=2020-02-28&shot=both&grp=1&dst=plyr&sort=sfg3&order=DESC'

url = 'http://stats.inpredictable.com/nba/ssnPlayerSplit.php?season=2022&pos=ALL&team=ALL&po=0&frdt=2019-10-22&todt=2020-02-28&shot=both&dst=plyr'

headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.116 Safari/537.36'}

#making request
response = requests.get(url,headers=headers)

response.status_code #Need 200 to show that you have downloaded HTML content from that page

#Check to see if correct HTML data is loaded
#response.content

soup = BeautifulSoup(response.content,'html.parser')

stat_table = soup.find_all(class_ = 'iptbl')
type(stat_table)
df = pd.read_html(str(stat_table))

#data processing
df = df[0]

df = df[1:]

df.rename(columns = {'Unnamed: 0_level_0':'', 
                      'Unnamed: 1_level_0':'',
                      'Unnamed: 2_level_0':'',
                      'Unnamed: 3_level_0':'',
                      'Unnamed: 4_level_0':'',
                     'All Field Goals (sortable)':'All Field Goals',
                     'All Field Goals (sortable)':'All Field Goals',
                     'All Field Goals (sortable)':'All Field Goals',
                     'All Field Goals (sortable)':'All Field Goals'
                     },inplace=True)
df.columns

df.dtypes

#transforming data
df['','Gms'] = df['','Gms'].astype('int') 
df['All Field Goals', 'Grbg'] = df['All Field Goals', 'Grbg'].astype('int')
df['All Field Goals', 'Nrml'] = df['All Field Goals', 'Nrml'].astype('int')
df['All Field Goals', 'Cltch'] = df['All Field Goals', 'Cltch'].astype('int')
df['All Field Goals', 'Cltch2'] = df['All Field Goals', 'Cltch2'].astype('int')

df['% of Player Shots', 'Grbg'] = df['% of Player Shots', 'Grbg'].str.rstrip('%').astype('float') / 100.0
df['% of Player Shots', 'Nrml'] = df['% of Player Shots', 'Nrml'].str.rstrip('%').astype('float') / 100.0
df['% of Player Shots', 'Cltch'] = df['% of Player Shots', 'Cltch'].str.rstrip('%').astype('float') / 100.0
df['% of Player Shots', 'Cltch2'] = df['% of Player Shots', 'Cltch2'].str.rstrip('%').astype('float') / 100.0
df['Effective Field Goal %','Grbg'] = df['Effective Field Goal %', 'Grbg'].str.rstrip('%').astype('float') / 100.0
df['Effective Field Goal %','Nrml'] = df['Effective Field Goal %', 'Nrml'].str.rstrip('%').astype('float') / 100.0
df['Effective Field Goal %','Cltch'] = df['Effective Field Goal %', 'Cltch'].str.rstrip('%').astype('float') / 100.0
df['Effective Field Goal %','Cltch2'] = df['Effective Field Goal %', 'Cltch2'].str.rstrip('%').astype('float') / 100.0

#checking data types after transformation
df.dtypes

#EDA to visualize Field Goals and % in Clutch
plt.figure(figsize=(20,10))
sns.scatterplot(x= df['All Field Goals']['Cltch'],y = df['Effective Field Goal %']['Cltch'],hue=df['']['Pos'])
plt.xlabel("# of Field Goals in Clutch")
plt.ylabel("% in Clutch")
plt.title('% in Clutch vs # of FGs in Clutch by Player')

for i in range(1, len(df) + 1):
    x = df['All Field Goals']['Cltch'][i]
    y = df['Effective Field Goal %']['Cltch'][i]
    label = df['']['Player'][i]
    plt.text(x, y, label, ha='center', va='center', rotation=45)

plt.show()

#EDA to visualize Field Goals and % in Clutch2
plt.figure(figsize=(20,10))
sns.scatterplot(x=df['All Field Goals']['Cltch2'],y=df['Effective Field Goal %']['Cltch2'],hue=df['']['Pos'])
plt.xlabel("# of Field Goals in Clutch2")
plt.ylabel("% in Clutch2")

for i in range(1, len(df) + 1):
    x = df['All Field Goals']['Cltch2'][i]
    y = df['Effective Field Goal %']['Cltch2'][i]
    label = df['']['Player'][i]
    plt.text(x, y, label, ha='center', va='center', rotation=45)

plt.show()

#Correlation Matrix using Pearsons CE for linear relationships

#Create a numeric df for scaling
num_df = df.iloc[:,4:17]
num_df.index = df['']['Player']
num_df.head()

#scaling the data
#create two different dfs for clutch and clutch2 data
num_df2 = num_df[[('All Field Goals', 'Cltch2'),('Effective Field Goal %', 'Cltch2')]]
num_df3 = num_df[[('All Field Goals', 'Cltch'),('Effective Field Goal %', 'Cltch')]]

df_scaled_cltch2 = pd.DataFrame(normalize(num_df2),columns=num_df2.columns)
df_scaled_cltch2.index+=1
df_scaled_cltch2.index = df['']['Player']    
df_scaled_cltch2.head()

df_scaled_cltch = pd.DataFrame(normalize(num_df3),columns=num_df3.columns)
df_scaled_cltch.index+=1
df_scaled_cltch.index = df['']['Player']    
df_scaled_cltch.head()


'''
#scaling the data
df_scaled = pd.DataFrame(normalize(num_df),columns=num_df.columns)
df_scaled.index+=1
df_scaled.index = df['']['Player']    
df_scaled.head()
'''

##plotting a dendrogram from hierarchical clustering using ward method for linkage
plt.figure(figsize=(20,10))
plt.title('Dendrograms')
dend = shc.dendrogram(shc.linkage(df_scaled_cltch2,method='ward'),labels=df_scaled_cltch2.index)


##agglomerative clustering 
cluster = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')
cluster.fit_predict(df_scaled_cltch2)

plt.figure(figsize=(20, 10))  
#plt.xlim(-.00001,.00008)
#plt.ylim(-.0005,.00225)

sns.scatterplot(x=df_scaled_cltch2['All Field Goals']['Cltch2'], 
            y=df_scaled_cltch2['Effective Field Goal %']['Cltch2'],cmap='coolwarm',c=cluster.labels_)



plt.title('Scaled Clutch 2 Data')
plt.xlabel('% of Player Shots in Clutch2')
plt.ylabel('EFG% in Clutch2')


for i, name in enumerate(df['']['Player']):
    plt.annotate(name, (df_scaled_cltch2['All Field Goals']['Cltch2'][i], 
            df_scaled_cltch2['Effective Field Goal %']['Cltch2'][i]),rotation=45, ha='center')
'''
# Selectively annotate a subset of points
texts = []
for i, name in enumerate(df['']['Player']):
    x = df_scaled['% of Player Shots']['Cltch2'][i]
    y = df_scaled['Effective Field Goal %']['Cltch2'][i]
    texts.append(plt.text(x, y, name, ha='center', va='center', rotation=45))

# Adjust the positions of labels to minimize overlap
adjust_text(texts)
'''
plt.show()


#k-means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2,init = 'random', n_init = 10)
kmeans.fit(df_scaled_cltch2)
kmeans.cluster_centers_

#plotting elbow method for optimal # of clusters
plt.figure(figsize=(20, 10))

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(df_scaled_cltch2)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


#Figure shows cluster that minimizes wcss is 7
kmeans = KMeans(n_clusters = 7, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(df_scaled_cltch2)

#beginning of  the cluster numbering with 1 instead of 0
y_kmeans1=y_kmeans
y_kmeans1=y_kmeans+1

# Adding cluster to the Dataset1
df.loc[:, ('', 'cltch2_cluster')] = y_kmeans1

plt.figure(figsize=(20, 10))
sns.set_style('darkgrid')
p1 = sns.scatterplot(x=df['All Field Goals']['Cltch2'], y=df['Effective Field Goal %']['Cltch2'], palette='Dark2', hue=df['']['cltch2_cluster'])

for line in range(1, df.shape[0]):
    p1.text(x=df['All Field Goals']['Cltch2'][line]-1, y=df['Effective Field Goal %']['Cltch2'][line],
             s=df['']['Player'][line], horizontalalignment='left',
             size='medium', color='black')
     #adjust_text(texts,arrowprops=dict(arrowstyle='->', color='red'))
        
plt.xlabel("# of Field Goals in Clutch2")
plt.ylabel("% in Clutch2")
plt.title('K Means Clusters of Players # of Field Goals in the Clutch2 vs EFG% in Clutch 2')

#Mean of clusters
kmeans_mean_cluster = pd.DataFrame(round(df.groupby([('', 'cltch2_cluster')]).mean(), 1))
kmeans_mean_cluster
