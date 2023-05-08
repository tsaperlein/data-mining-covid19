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
grouped = df.groupby(['Entity', pd.Grouper(key='Date', freq='W-MON')]).agg({
    'Daily tests': 'sum',
    'Cases': 'max',
    'Deaths': 'max',
    'Population': 'max'
})

# get the first week of each country
first_week = grouped.groupby('Entity').head(1)

# calculate the new columns
# grouped['Cases/tests per week'] = grouped['Cases'].diff() / grouped['Daily tests']
grouped['Cases/tests per week'] = np.where(grouped['Daily tests'] != 0, grouped['Cases'].diff() / grouped['Daily tests'], 0)
grouped['Deaths/cases per week'] = grouped['Deaths'].diff() / grouped['Cases'].diff()
grouped['Tests/population per week'] = grouped['Daily tests'] / grouped['Population']

# remove the first week from the data(we don't have the previous week to calculate the difference)
grouped = grouped.drop(first_week.index)
# calculate the new columns for the first week
first_week['Cases/tests per week'] = first_week['Cases'] / first_week['Daily tests']
first_week['Deaths/cases per week'] = first_week['Deaths'] / first_week['Cases']
first_week['Tests/population per week'] = first_week['Daily tests'] / first_week['Population']

# add the first week to the start of the data
grouped = pd.concat([first_week, grouped])

# reset the index
grouped = grouped.reset_index()

new_df = grouped[['Entity', 'Cases/tests per week', 'Deaths/cases per week', 'Tests/population per week']]
# remove inf values (no cases in a week)
new_df = new_df[new_df['Deaths/cases per week'] != np.inf]
# remove nan values (no cases and no deaths in a week)
new_df = new_df.dropna()
# remove negative values
new_df.drop(new_df[new_df['Cases/tests per week'] < 0].index, inplace = True)
new_df.drop(new_df[new_df['Deaths/cases per week'] < 0].index, inplace = True)
stats = new_df.describe()
print(stats)

# Use one hot encoding to convert categorical data to numerical data
new_df = pd.get_dummies(new_df, columns=['Entity'])

new_df.to_csv('new_dataset.csv', index=False)


# --- CLUSTERING ---
# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(new_df.values)

# Elbow method to determine the number of clusters
# Instantiate the clustering model and visualizer
kmeans = KMeans(init='k-means++', n_init=10)
visualizer = KElbowVisualizer(kmeans, k=(2,10))
 
visualizer.fit(X)
visualizer.show()