import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Load the CSV file
df = pd.read_csv("data.csv")
df.info()
stats = df.describe()
print("\n \n")
print(stats)

# Print the number of missing values in each column of the DataFrame, sorted in descending order
NaNs_data_df = df.isnull().sum().sort_values(ascending=False)
print("\n \n")
print(NaNs_data_df)

# Filter out inaccurate values
negative_tests = (df['Daily tests'] < 0).sum()
print("\n \nNegative daily tests:", negative_tests)
df = df[df['Daily tests'] >= 0]


# --- Fill missing values in columns
# Group by Entity and fill missing values
df = df.groupby(df['Entity']).apply(lambda x: x.fillna(method='ffill'))
df = df.groupby(df['Entity']).apply(lambda x: x.fillna(method='bfill'))
# Filter out rows with no cases
df = df[df.Cases > 0]



# --- MAP ------------------------------------
def lng_lat_to_pixels(lng, lat):    
    lng_rad = lng * np.pi / 180
    lat_rad = lat * np.pi / 180
    x = (256/(2*np.pi))*(lng_rad + np.pi)
    y = (256/(2*np.pi))*(np.log(np.tan(np.pi/4 + lat_rad/2)))
    return (x, y)

# Group by "Entity" and find the maximum value of "Deaths" for each group
max_deaths_index = df.groupby('Entity')['Deaths'].idxmax()
max_deaths = df.loc[max_deaths_index]

px, py = lng_lat_to_pixels(max_deaths['Longitude'], max_deaths['Latitude'])
sizes = max_deaths['Deaths'].values

print(px.min(), py.min(), px.max(), py.max())

# Plot the points
plt.figure(figsize=(12, 10))
im = plt.imread("map.jpg")
plt.imshow(im, extent=[52.37, 255.58, -31.92, 61.31])
plt.axis('equal')
# plt.axis('off')
plt.gca().set_facecolor('white')
_ = plt.scatter(px, py, s=0.001*sizes, color='black')
plt.show()
# --------------------------------------------


# --- Plot/save boxplots for each column
columns = ['Entity', 'Continent', 'Date', 'Daily tests', 'Cases', 'Deaths']
data = df.drop(columns, axis=1).drop_duplicates()
fig, axs = plt.subplots(3, 3, figsize=(12, 8))
axs = axs.flatten()
for i, col in enumerate(data.columns):
    axs[i].boxplot(data[col])
    axs[i].set_title(col)
plt.tight_layout()
plt.savefig("images/boxplots.png", bbox_inches='tight')
plt.close()

# --- Plot histograms for each column
columns = ['Date', 'Daily tests', 'Cases', 'Deaths']
plt.figure(figsize=(16, 9))
df.drop(columns, axis=1).drop_duplicates().hist(bins=15, figsize=(16, 9), rwidth=0.8)
plt.savefig("images/histograms.png", bbox_inches='tight')
plt.close()

# --- Plot heatmap for correlation between columns
# Keep the last line (date) for each country and drop unused columns
df_last = df.groupby(df['Entity']).tail(1).drop(['Entity', 'Date', 'Continent'], axis=1)
plt.figure(figsize=(12, 8))
sns.heatmap(df_last.corr(), annot=True, cmap=plt.cm.Reds)
plt.savefig("images/correlation-heatmap.png", bbox_inches='tight')
plt.close()

# --- Plot case and death curves for selected countries
""" # Target countries and target dates
countries = ['Algeria', 'Bahrain', 'Ethiopia', 'Ghana', 'Kenya', 'Morocco', 'Nigeria', 'Senegal', 'Tunisia']
df_temp = df.loc[df['Date'] > '2020-02-25']

# Plot case and death curves
for output_variable in ['Cases', 'Deaths']:
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, grp in df_temp[df_temp['Entity'].isin(countries)].groupby(df['Entity']):
        ax = grp.plot(ax=ax, kind='line', x='Date', y=output_variable, label=key)
    plt.legend(loc='best')
    plt.xticks(rotation=90)
    plt.ylabel(output_variable)
    plt.savefig(f"images/Curves/{output_variable}-curves.png", bbox_inches='tight')
    plt.close() """
    
# --- Plot feature-output-variable distributions for each column
# Scatter plots readability: remove outliers in all comuns except in the column 'Continent'
df_last = df.groupby(df['Entity']).tail(1).drop(['Entity', 'Date'], axis=1)
column_continent = df_last[['Continent']]
df_last = df_last.drop('Continent', axis=1)
df_last = column_continent.join(df_last[(np.abs(zscore(df_last)) < 3).all(axis=1)])

# Plot feature-output-variable distributions for each column
for column in df_last.columns.drop(['Cases', 'Deaths']):
    fig, ax = plt.subplots(ncols=2, figsize=(14, 4))
    df_last.plot.scatter(x=column, y='Cases', ax=ax[0])
    df_last.plot.scatter(x=column, y='Deaths', ax=ax[1])
    if column == 'Continent':
        fig.autofmt_xdate(rotation=90)
    file_name = column.replace("/", "-")
    plt.savefig(f"images/Scatter/{file_name}-scatter.png", bbox_inches='tight')
    plt.close()
#scatter deaths-cases
fig, ax = plt.subplots(figsize=(10, 6))
df_last.plot.scatter(x='Cases', y='Deaths', ax=ax)
plt.savefig(f"images/Scatter/deaths-cases-scatter.png", bbox_inches='tight')
plt.close()