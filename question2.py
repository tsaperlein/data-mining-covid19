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
from gap_statistic import OptimalK
from sklearn.metrics import calinski_harabasz_score

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
# - Elbow Method before standardization
kmeans = KMeans(n_init=10)
visualizer = KElbowVisualizer(kmeans, numeric_only=True)
visualizer.fit(data_values.values)
visualizer.show(outpath="elbow_method_1.png")
visualizer.show()

k_elbow_1 = visualizer.elbow_value_

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(data_values.values)

# - Elbow Method after standardization
kmeans = KMeans(n_init=10)
visualizer = KElbowVisualizer(kmeans, numeric_only=True)
visualizer.fit(X)

# Save the visualization as a PNG file
visualizer.show(outpath="elbow_method_2.png")
visualizer.show()

# Get the optimal number of clusters suggested by the elbow method
k_elbow_2 = visualizer.elbow_value_
# --------------------------------------------

# - Sihouette Method
from yellowbrick.cluster import  SilhouetteVisualizer
from sklearn.metrics import silhouette_score

silhouette = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters = k, n_init=10).fit(X)
    preds = kmeans.fit_predict(X)
    silhouette.append(silhouette_score(X, preds))
    
plt.plot(range(2, 11), silhouette, 'bo-')
plt.xlabel('k')
plt.ylabel('Silhouette score')
plt.title('Silhouette Method For Optimal k')
plt.savefig('silhouette_method.png')
plt.show()

# Get the optimal number of clusters suggested by the Silhouette method
k_silhouette = silhouette.index(max(silhouette)) + 2
# --------------------------------------------

# Print the results of the three methods
print("\n- Elbow method (before standardization) suggests", k_elbow_1, "clusters")
print("\n- Elbow method (after standardization) suggests", k_elbow_2, "clusters")
print("\n- Silhouette method suggests", k_silhouette, "clusters")

# Compare the results of the two methods and choose the best value of k
if k_elbow_2 == k_silhouette:
    print("\nThe optimal number of clusters is:", k_elbow_2)
else:
    print('\nBecause the two methods suggest different values for k, we will go with the value suggested by the Silhouette method, which is', k_silhouette)
    # Other methods to find the optimal number of clusters (Gap statistic method, Calinski-Harabasz index method)
    '''# - Gap statistic Method
    optimalK = OptimalK(parallel_backend='joblib')
    n_clusters = optimalK(X, cluster_array=np.arange(2, 11))

    plt.plot(optimalK.gap_df.n_clusters, optimalK.gap_df.gap_value, linestyle='--', marker='o', color='b')
    plt.xlabel('Number of clusters')
    plt.ylabel('Gap value')
    plt.title('Gap statistic for KMeans clustering')
    plt.savefig('gap_statistic_method.png')
    plt.show()
    
    # Get the optimal number of clusters suggested by the gap statistic method
    k_gap = n_clusters
    
    # - Calinski-Harabasz index method
    ch_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        ch_scores.append(calinski_harabasz_score(X, kmeans.labels_))

    plt.plot(range(2, 11), ch_scores, linestyle='--', marker='o', color='b')
    plt.xlabel('Number of clusters')
    plt.ylabel('Calinski-Harabasz score')
    plt.title('Calinski-Harabasz index for KMeans clustering')
    plt.savefig('calinski_harabasz_index_method.png')
    plt.show()
    
    # Get the optimal number of clusters suggested by the Calinski-Harabasz index method
    k_calinski = ch_scores.index(max(ch_scores)) + 2
    
    print("\nGap statistic method suggests", k_gap, "clusters")
    print("Calinski-Harabasz index method suggests", k_calinski, "clusters")'''
    
# --- Silhouette Score ---
for k in range(2,6):
    kmeans = KMeans(n_clusters=k,n_init=10)
    visualizer = SilhouetteVisualizer(kmeans)
    visualizer.fit(X)
    visualizer.show(outpath="sil_score_" + str(k) + ".png")
    visualizer.show()
# --------------------------------------------

# --- K-Means Clustering ---
from sklearn.cluster import KMeans

fig = plt.figure()

# Perform K-Means clustering
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
    clusters = kmeans.fit_predict(X)
    ax = fig.add_subplot(2, 2, k-1)
    ax.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
    ax.set_title('k = ' + str(k))
    ax.set_xlabel('Cases/tests')
    ax.set_ylabel('Deaths/cases')
    # Change the color of the cluster centers
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='o', c='r')
    # Add the cluster centers as annotations
    for i, c in enumerate(kmeans.cluster_centers_):
        ax.annotate(i+1, (c[0], c[1]), fontsize=20, color='r')

# Plot the figure
fig.text(0.5, 0.04, 'Cluster centers are marked in red color', ha='center', va='center', fontsize=15, color='r')  
plt.tight_layout()
plt.gcf().set_size_inches(13, 7)
plt.savefig('kmeans_clustering.png')
plt.show()
# --------------------------------------------

# --- Clustering Visualization ---
for k in range(2,6):
    kmeans = KMeans(n_clusters=k,n_init=10)
    kmeans.fit(X)

    labels = kmeans.labels_
    
    plt.scatter(new_df['Cases/tests'], new_df['Deaths/cases'], c=labels, cmap='viridis', alpha=0.5, edgecolors='b')

    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, linewidths=1, color='black')
    
    # Add labels to the plot
    plt.xlabel('Cases/tests')
    plt.ylabel('Deaths/cases')
    plt.title('K-Means Clustering (k = ' + str(k) + ')')

    # Show/save the plot
    plt.savefig('kmeans_clustering_' + str(k) + '.png')
    plt.show()
# --------------------------------------------