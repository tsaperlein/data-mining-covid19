import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("data.csv")
df.info()
stats = df.describe()
print(stats)

NaNs_data_df = df.isnull().sum().sort_values(ascending=False)
print(NaNs_data_df)

""" for column in df.columns:
    plt.figure()  # Create a new figure for each plot
    sns.histplot(df[column])
    plt.title(column)  # Add the column name as the title
    plt.savefig(f"{column}.png") """

