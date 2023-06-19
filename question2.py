import pandas as pd
import numpy as np
import geopandas as gpd
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


np.random.seed(0)

# Load the CSV file
df = pd.read_csv("modified_dataframe.csv")


# group the data by country and aggregate the columns
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

new_df.to_csv('new_dataframe.csv', index=False)




# ----------------------------------------------------
# -------------------- CLUSTERING --------------------
data_values = new_df.drop('Entity', axis=1)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(data_values.values)
X = pd.DataFrame(X, index=data_values.index, columns=data_values.columns)

# --------- Elbow Method
kmeans = KMeans(n_init=10)
visualizer = KElbowVisualizer(kmeans, numeric_only=True)
visualizer.fit(X)
visualizer.show(outpath="q2/elbow_method.png")
plt.close()

# Get the optimal number of clusters suggested by the elbow method
k_elbow = visualizer.elbow_value_


# ---------- Sihouette Method
silhouette = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters = k, n_init=10).fit(X)
    preds = kmeans.fit_predict(X)
    silhouette.append(silhouette_score(X, preds))
    
plt.plot(range(2, 11), silhouette, 'bo-')
plt.xlabel('k')
plt.ylabel('Silhouette score')
plt.title('Silhouette Method For Optimal k')
plt.savefig('q2/silhouette_method.png')

# Get the optimal number of clusters suggested by the Silhouette method
k_silhouette = silhouette.index(max(silhouette)) + 2


# Print the results of the two methods
print("\n- Elbow method suggests", k_elbow, "clusters")
print("- Silhouette method suggests", k_silhouette, "clusters")
    

# ------------ Silhouette Score 
plt.figure(figsize=(10,  7))
scores = {}
for k in range(2, 6):
    plt.subplot(2, 2, k - 1)
    kmeans = KMeans(n_clusters=k,n_init=10)
    visualizer = SilhouetteVisualizer(kmeans)
    visualizer.fit(X)
    scores[k] = visualizer.silhouette_score_
    plt.title(f'clusters: {k} score: {visualizer.silhouette_score_}')
plt.savefig('q2/silhouette_score.png')


# -------------- PCA

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
plt.savefig('q2/pca.png')




# ---------------------------------------------------
# --------------- Cluster Analysis ------------------

clustered_df = new_df.copy()
clustered_df["Cluster"] = clusters

# Print the countries in each cluster
for cluster in range(0, 4):
    print(f"\nCluster {cluster} countries:")
    print(clustered_df[clustered_df["Cluster"] == cluster]["Entity"].values)


# Violin Plot
i=0
for metric in clustered_df.columns[1:-2]:
    i+=1
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=clustered_df['Cluster'], y=clustered_df[metric])
    plt.savefig(f"q2/violin_{i}.png")
    plt.close()

print("\n\n")
print(clustered_df[clustered_df["Cluster"] == 3])



# the unused columns
country_info = df.groupby(['Entity']).agg({
    "Entity": "first",
    "Latitude": "mean",
    "Longitude": "mean",
    "Average temperature per year": "mean",
    "Hospital beds per 1000 people": "mean",
    "Medical doctors per 1000 people": "mean",
    "GDP/Capita": "mean",
    "Median age": "mean",
    "Population aged 65 and over (%)": "mean",
    "Population": "mean"
})
clustered_info = country_info.copy()
clustered_info["Cluster"] = clusters



# Violin Plot
j=0
for info in clustered_info.columns[3:-2]:
    j+=1
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=clustered_info['Cluster'], y=clustered_info[info])
    plt.savefig(f"q2/violin_info{j}.png")

# Population mean value
population_0 = clustered_info[clustered_info["Cluster"] == 0]["Population"].mean()
population_1 = clustered_info[clustered_info["Cluster"] == 1]["Population"].mean()
population_2 = clustered_info[clustered_info["Cluster"] == 2]["Population"].mean()
population_3 = clustered_info[clustered_info["Cluster"] == 3]["Population"].mean()
print("\n\n")
print(f"Population mean value for cluster 0: {population_0}")
print(f"Population mean value for cluster 1: {population_1}")
print(f"Population mean value for cluster 2: {population_2}")
print(f"Population mean value for cluster 3: {population_3}")




# --- MAP ---
# From GeoPandas, our world map data
worldmap = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Creating axes and plotting world map
fig, ax = plt.subplots(figsize=(12, 6))
worldmap.plot(color="lightgrey", ax=ax)

colors = {0: "red", 1: "blue", 2: "green", 3: "yellow"}

# Plotting our Enitites with different colors depending on the cluster
for country in clustered_info['Entity']:
    x = df[df['Entity'] == country]['Longitude']
    y = df[df['Entity'] == country]['Latitude']
    cluster = clustered_info[clustered_info['Entity'] == country]['Cluster'].values[0]
    population = df[df['Entity'] == country]['Population'].values[0]
    ax.scatter(x, y, color=colors[cluster], alpha=1, s=pow(population/1000000, 0.6))
    
# Show colors legend
for cluster, color in colors.items():
    ax.scatter([], [], c=color, alpha=1, s=15, label=f"Cluster {cluster}")

ax.legend(scatterpoints=1, frameon=True, labelspacing=1, loc='lower left', facecolor='gray', fontsize=12, labelcolor='white');

# Creating axis limits and title
plt.xlim([-180, 180])
plt.ylim([-90, 90])

plt.title("Countries by cluster")
plt.savefig("q2/countries_by_cluster.png", dpi=300, bbox_inches='tight')