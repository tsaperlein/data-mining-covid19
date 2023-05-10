import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.datasets import make_blobs
from yellowbrick.cluster import  SilhouetteVisualizer
from sklearn.metrics import silhouette_score

# Load the CSV file
df = pd.read_csv("data.csv")

# Filter out inaccurate values in "Daily tests" 
df.drop(df[df['Daily tests'] < 0].index, inplace = True)

# --- Fill missing values in columns ---
# Fill missing values in "Daily tests" column
# df['Daily tests'] = df['Daily tests'].groupby(df['Entity']).apply(lambda x: x.fillna(method='ffill'))
# df['Daily tests'] = df['Daily tests'].groupby(df['Entity']).apply(lambda x: x.fillna(method='bfill'))
df['Daily tests'] = df['Daily tests'].groupby(df['Entity']).apply(lambda x: x.fillna(method='ffill')).reset_index(drop=True)
df['Daily tests'] = df['Daily tests'].groupby(df['Entity']).apply(lambda x: x.fillna(method='bfill')).reset_index(drop=True)
# Fill missing values in "Cases" column with 0
df['Cases'] = df['Cases'].fillna(0)
# Fill missing values in "Deaths" column with 0
df['Deaths'] = df['Deaths'].fillna(0)

# group the data by country and week, and aggregate the columns
grouped = df.groupby(['Entity']).agg({
    'Daily tests': 'sum',
    'Cases': 'max',
    'Deaths': 'max',
    'Population': 'mean'
})

# calculate the new columns
grouped['Cases/tests'] = grouped['Cases'] / grouped['Daily tests']
grouped['Deaths/cases'] = grouped['Deaths'] / grouped['Cases']
grouped['Tests/population'] = grouped['Daily tests'] / grouped['Population']

# reset the index
grouped = grouped.reset_index()

new_df = grouped[['Entity', 'Cases/tests', 'Deaths/cases', 'Tests/population']]

stats = new_df.describe()
print(stats)

new_df.to_csv('new_dataset.csv', index=False)

# --- CLUSTERING ---
data_values = new_df.drop('Entity', axis=1)

# -- Find the optimal number of clusters --
# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(data_values.values)

# - Elbow Method
kmeans = KMeans(n_init=10)
visualizer = KElbowVisualizer(kmeans, numeric_only=True)
visualizer.fit(X)
visualizer.show(outpath="img/elbow_method.png")
plt.close()

# Get the optimal number of clusters suggested by the elbow method
k_elbow = visualizer.elbow_value_
# --------------------------------------------

# - Sihouette Method
silhouette = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters = k, n_init=10).fit(X)
    preds = kmeans.fit_predict(X)
    silhouette.append(silhouette_score(X, preds))
    
plt.plot(range(2, 11), silhouette, 'bo-')
plt.xlabel('k')
plt.ylabel('Silhouette score')
plt.title('Silhouette Method For Optimal k')
plt.savefig('img/silhouette_method.png')

# Get the optimal number of clusters suggested by the Silhouette method
k_silhouette = silhouette.index(max(silhouette)) + 2
# --------------------------------------------

# Print the results of the three methods
print("\n- Elbow method suggests", k_elbow, "clusters")
print("- Silhouette method suggests", k_silhouette, "clusters")
    
# --- Silhouette Score ---
plt.figure(figsize=(10,  7))
scores = {}
for k in range(2, 6):
    plt.subplot(2, 2, k - 1)
    kmeans = KMeans(n_clusters=k,n_init=10)
    visualizer = SilhouetteVisualizer(kmeans)
    visualizer.fit(X)
    scores[k] = visualizer.silhouette_score_
    plt.title(f'clusters: {k} score: {visualizer.silhouette_score_}')
plt.savefig('img/silhouette_score.png')
# --------------------------------------------

# --- K-Means Clustering ---
fig = plt.figure()

# Perform K-Means clustering
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
    clusters = kmeans.fit(X)
    ax = fig.add_subplot(2, 2, k-1)
    ax.scatter(X[:, 0], X[:, 1], c=clusters.labels_, cmap='viridis')
    ax.set_title('k = ' + str(k))
    ax.set_xlabel('Cases/tests')
    ax.set_ylabel('Deaths/cases')
    ax.scatter(clusters.cluster_centers_[:, 0], clusters.cluster_centers_[:, 1], c='red', s=50)

# Plot the figure
fig.text(0.5, 0.04, 'Cluster centers are marked in red color', ha='center', va='center', fontsize=15, color='r')  
plt.tight_layout()
plt.gcf().set_size_inches(13, 7)
plt.savefig('img/kmeans_clustering.png')
# --------------------------------------------