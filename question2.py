import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer

# Load the CSV file
df = pd.read_csv("data.csv")

# Filter out inaccurate values in "Daily tests" 
df.drop(df[df['Daily tests'] < 0].index, inplace = True)

# --- Fill missing values in columns
# Fill missing values in "Daily tests" column
# df['Daily tests'] = df['Daily tests'].groupby(df['Entity']).apply(lambda x: x.fillna(method='ffill'))
# df['Daily tests'] = df['Daily tests'].groupby(df['Entity']).apply(lambda x: x.fillna(method='bfill'))
df['Daily tests'] = df['Daily tests'].groupby(df['Entity']).apply(lambda x: x.fillna(method='ffill')).reset_index(drop=True)
df['Daily tests'] = df['Daily tests'].groupby(df['Entity']).apply(lambda x: x.fillna(method='bfill')).reset_index(drop=True)
# Fill missing values in "Cases" column with 0
df['Cases'] = df['Cases'].fillna(0)
# Fill missing values in "Deaths" column with 0
df['Deaths'] = df['Deaths'].fillna(0)

# convert date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# group the data by country and week, and aggregate the columns
grouped = df.groupby(['Entity']).agg({
    'Daily tests': 'sum',
    'Cases': 'max',
    'Deaths': 'max',
    'Population': 'mean'
})


# calculate the new columns
# grouped['Cases/tests'] = grouped['Cases'] / grouped['Daily tests']
grouped['Cases/tests per week'] = np.where(grouped['Daily tests'] != 0, grouped['Cases'].diff() / grouped['Daily tests'], 0)
grouped['Deaths/cases'] = grouped['Deaths'] / grouped['Cases']
grouped['Tests/population'] = grouped['Daily tests'] / grouped['Population']

# reset the index
grouped = grouped.reset_index()

new_df = grouped[['Entity', 'Cases/tests', 'Deaths/cases', 'Tests/population']]

stats = new_df.describe()
print(stats)


new_df.to_csv('new_dataset.csv', index=False)


# --- CLUSTERING ---
# Standardize the data
data_values = new_df.drop('Entity', axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(data_values.values)

# ELBOW METHOD
kmeans = KMeans(init='k-means++', n_init=10)
visualizer = KElbowVisualizer(kmeans, k=(2,10))
 
visualizer.fit(X)
visualizer.show()

# Get the optimal number of clusters suggested by the elbow method
k_elbow = visualizer.elbow_value_

# SILHOUETTE METHOD
from yellowbrick.cluster import  SilhouetteVisualizer
from sklearn.metrics import silhouette_score

silhouette = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters = k, n_init=10).fit(X)
    preds = kmeans.fit_predict(X)
    silhouette.append(silhouette_score(X, preds))
    
plt.plot(range(2, 10), silhouette, 'bo-')
plt.xlabel('k')
plt.ylabel('Silhouette score')
plt.title('Silhouette Method For Optimal k')
plt.savefig('silhouette.png')

# Get the optimal number of clusters suggested by the Silhouette method
k_silhouette = silhouette.index(max(silhouette)) + 2

print("\nElbow method suggests", k_elbow, "clusters")
print("Silhouette method suggests", k_silhouette, "clusters")

# Compare the results of the two methods and choose the best value of k
if k_elbow == k_silhouette:
    print("\nThe optimal number of clusters is:", k_elbow)