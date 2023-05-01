import pandas as pd

# Load the CSV file
df = pd.read_csv("data.csv")

# Filter out inaccurate values in "Daily tests" 
df.drop(df[df['Daily tests'] < 0].index, inplace = True)

# --- Fill missing values in columns
# Fill missing values in "Daily tests" column
df['Daily tests'] = df['Daily tests'].groupby(df['Entity']).apply(lambda x: x.fillna(method='ffill'))
df['Daily tests'] = df['Daily tests'].groupby(df['Entity']).apply(lambda x: x.fillna(method='bfill'))
# Fill missing values in "Cases" column with 0
df['Cases'] = df['Cases'].fillna(0)
# Fill missing values in "Deaths" column with 0
df['Deaths'] = df['Deaths'].fillna(0)

inf = (df['Daily tests'] == 0).sum()
print("\n \nInfinite daily tests:", inf)

# convert date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# group the data by country and week, and aggregate the columns
grouped = df.groupby(['Entity', pd.Grouper(key='Date', freq='W-MON')]).agg({
    'Daily tests': 'sum',
    'Cases': 'max',
    'Deaths': 'max',
    'Population': 'max'
})

# calculate the new columns
grouped['Cases/tests per week'] = grouped['Cases'].diff() / grouped['Daily tests']
grouped['Deaths/cases per week'] = grouped['Deaths'].diff() / grouped['Cases'].diff()
grouped['Tests/population per week'] = grouped['Daily tests'] / grouped['Population']

# reset the index to make the country and week columns regular columns
grouped = grouped.reset_index()

# select the desired columns and save to a new csv file
new_df = grouped[['Entity', 'Cases/tests per week', 'Deaths/cases per week', 'Tests/population per week']]
new_df.to_csv('new_dataset.csv', index=False)