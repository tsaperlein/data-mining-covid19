import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("data.csv")
df.info()
stats = df.describe()
print(stats)

# Print the number of missing values in each column of the DataFrame, sorted in descending order
NaNs_data_df = df.isnull().sum().sort_values(ascending=False)
print(NaNs_data_df)

# --- Plot/save figures for each column
# for column in df.columns:
#     plt.figure()  # Create a new figure for each plot
#     sns.histplot(df[column])
#     plt.title(column)  # Add the column name as the title
#     plt.savefig(f"{column}.png")

# --- Fill missing values in columns
# Group by Entity and fill missing values
df = df.groupby(df['Entity']).apply(lambda x: x.fillna(method='ffill'))
df = df.groupby(df['Entity']).apply(lambda x: x.fillna(method='bfill'))

# Filter out rows with no cases
df = df[df.Cases > 0]

print(df.isnull().sum())

# --- Plot/save boxplots for each column
for column in df.drop(['Entity', 'Continent', 'Date', 'Daily tests', 'Cases', 'Deaths'], axis=1):
    plt.figure()
    df.boxplot([column])