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
from sklearn.decomposition import PCA
# import plotly as py
# import plotly.graph_objs as go
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

np.random.seed(0)

# Load the CSV file
df = pd.read_csv("modified_dataset.csv")


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
grouped['Cases/population'] = grouped['Cases'] / grouped['Population']
grouped['Deaths/population'] = grouped['Deaths'] / grouped['Population']

# reset the index
grouped = grouped.reset_index()

new_df = grouped[['Entity', 'Cases/tests', 'Deaths/cases', 'Tests/population', 'Cases/population', 'Deaths/population']]

stats = new_df.describe()
print(stats)

new_df.to_csv('new_dataset.csv', index=False)

# --- CLUSTERING ---
data_values = new_df.drop('Entity', axis=1)

# -- Find the optimal number of clusters --
# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(data_values.values)
X = pd.DataFrame(X, index=data_values.index, columns=data_values.columns)

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
""" fig = plt.figure()

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
plt.savefig('img/kmeans_clustering.png') """
# --------------------------------------------

kmeans = KMeans(n_clusters=4, n_init=10).fit(X)
clusters = kmeans.labels_
X['Cluster'] = clusters

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(x=X_pca[:,0], y=X_pca[:,1], c=X['Cluster'], 
            edgecolor='k', s=100, alpha=0.5, cmap='viridis')
#show centroids
# plt.scatter(x=kmeans.cluster_centers_[:,0], y=kmeans.cluster_centers_[:,1],
#             s=100, c='red', label='centroids')
plt.grid(visible=True)
plt.savefig('img/pca.png')

""" PCs_2d = pd.DataFrame(pca.fit_transform(X.drop(["Cluster"], axis=1)))
PCs_2d.columns = ["PC1_2d", "PC2_2d"]

X = pd.concat([X,PCs_2d], axis=1, join='inner')

cluster0 = X[X["Cluster"] == 0]
cluster1 = X[X["Cluster"] == 1]
cluster2 = X[X["Cluster"] == 2]


#trace1 is for 'Cluster 0'
trace1 = go.Scatter(
                    x = cluster0["PC1_2d"],
                    y = cluster0["PC2_2d"],
                    mode = "markers",
                    name = "Cluster 0",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text = None)

#trace2 is for 'Cluster 1'
trace2 = go.Scatter(
                    x = cluster1["PC1_2d"],
                    y = cluster1["PC2_2d"],
                    mode = "markers",
                    name = "Cluster 1",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text = None)

#trace3 is for 'Cluster 2'
trace3 = go.Scatter(
                    x = cluster2["PC1_2d"],
                    y = cluster2["PC2_2d"],
                    mode = "markers",
                    name = "Cluster 2",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text = None)

data = [trace1, trace2, trace3]

title = "Visualizing Clusters in Two Dimensions Using PCA"

layout = dict(title = title,
              xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)

iplot(fig) """
