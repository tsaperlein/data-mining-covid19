import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# --- Load the CSV file ---
df = pd.read_csv("data.csv")
df.info()
stats = df.describe()
print("\n \n")
print(stats)

# Print the number of missing values in each column of the DataFrame, sorted in descending order
NaNs_data_df = df.isnull().sum().sort_values(ascending=False)
print("\n \n")
print(NaNs_data_df)

# Filter out inaccurate values in "Daily tests" 
df.drop(df[df['Daily tests'] < 0].index, inplace = True)
# --------------------------------------------


# Drop countries with too many missing values in "Daily tests" column
country_nan_percentage = df.groupby("Entity")["Daily tests"].apply(lambda x: x.isna().mean() * 100)
threshold = 75      # 75% missing values
countries_to_drop = country_nan_percentage[country_nan_percentage > threshold].index

# print the countries to drop
print("\n \n")
print(countries_to_drop)
df = df[~df["Entity"].isin(countries_to_drop)].reset_index(drop=True)

# --- Fill missing values in columns ---
# Fill missing values in "Daily tests" column
df['Daily tests'] = df['Daily tests'].groupby(df['Entity']).apply(lambda x: x.fillna(method='ffill')).reset_index(drop=True)
df['Daily tests'] = df['Daily tests'].groupby(df['Entity']).apply(lambda x: x.fillna(method='bfill')).reset_index(drop=True)
# Fill missing values in "Cases" column with 0
df['Cases'] = df['Cases'].fillna(0)
# Fill missing values in "Deaths" column with 0
df['Deaths'] = df['Deaths'].fillna(0)
# --------------------------------------------

# Save df to a new csv file
df.to_csv('modified_dataframe.csv', index=False)

# --- TIMELINE ---
# -- Plot the number of cases and deaths passing through time
# Convert date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Create a new dataframe with the required data
data = df[['Date', 'Entity', 'Cases', 'Deaths']]
data = data.groupby('Date').sum().reset_index()

# Create figure and axes
fig, ax = plt.subplots()

# Plot cases and deaths for all entities
ax.plot(data['Date'], data['Cases'], label='Cases')
ax.plot(data['Date'], data['Deaths'], label='Deaths')

# Set x and y labels
ax.set_xlabel('Month')
ax.set_ylabel('Number')

# Format x-axis labels
month_format = mpl.dates.DateFormatter('%b')
ax.xaxis.set_major_formatter(month_format)
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))

# Set y-axis limits
ax.set_ylim(bottom=1)
ax.set_yscale('log')

# Add legend
ax.legend()

# Save plot
plt.savefig('images/cases_deaths_diagram.png', dpi=300, bbox_inches='tight')
# --------------------------------------------

# --- MAP ---
# Find the range of longitude and latitude in the data
def lng_lat_to_pixels(lng, lat):    
    lng_rad = lng * np.pi / 180
    lat_rad = lat * np.pi / 180
    x = (256/(2*np.pi))*(lng_rad + np.pi)
    y = (256/(2*np.pi))*(np.log(np.tan(np.pi/4 + lat_rad/2)))
    return (x, y)

# Group by "Entity" and find the maximum value of "Deaths" for each group
max_deaths_index = df.groupby(df['Entity'])['Deaths'].idxmax()
max_deaths = df.loc[max_deaths_index]

px, py = lng_lat_to_pixels(max_deaths['Longitude'], max_deaths['Latitude'])
sizes = max_deaths['Deaths'].values / max_deaths['Population'].values
extent = [px.min(), px.max(), py.min(), py.max()] 

# Plot the points
plt.figure(figsize=(12, 8))
plt.axis('equal')
plt.gca().set_facecolor('gray')
_ = plt.scatter(px, py, s=5000*sizes, color='black')
plt.savefig('images/deaths_population_map.png', dpi=300, bbox_inches='tight')
# --------------------------------------------


# --- Plot/save boxplots for each column ---
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
# --------------------------------------------

# --- Plot histograms for each column ---
columns = ['Date', 'Daily tests', 'Cases', 'Deaths']
plt.figure(figsize=(16, 9))
df.drop(columns, axis=1).drop_duplicates().hist(bins=15, figsize=(16, 9), rwidth=0.8)
plt.savefig("images/histograms.png", bbox_inches='tight')
plt.close()
# --------------------------------------------

# --- Plot heatmap for correlation between columns ---
# Keep the last line (date) for each country and drop unused columns
df_last = df.groupby(df['Entity']).tail(1).drop(['Entity', 'Date', 'Continent'], axis=1)
plt.figure(figsize=(12, 8))
sns.heatmap(df_last.corr(), annot=True, cmap=plt.cm.Reds)
plt.savefig("images/correlation-heatmap.png", bbox_inches='tight')
plt.close()
# --------------------------------------------
    
# --- Plot feature-output-variable distributions for each column ---
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
# --------------------------------------------
    
# --- Scatter deaths-cases ---
fig, ax = plt.subplots(figsize=(10, 6))
df_last.plot.scatter(x='Cases', y='Deaths', ax=ax)
plt.savefig(f"images/Scatter/deaths-cases-scatter.png", bbox_inches='tight')
plt.close()