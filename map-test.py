import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy as crt
import cartopy.crs as ccrs
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load your data into a pandas dataframe
df = pd.read_csv('data.csv')

# Group by "Entity" and find the maximum value of "Deaths" for each group
max_deaths_index = df.groupby(df['Entity'])['Deaths'].idxmax()
max_deaths = df.loc[max_deaths_index]

# Convert longitude and latitude to pixels
def lng_lat_to_pixels(lng, lat):    
    lng_rad = lng * np.pi / 180
    lat_rad = lat * np.pi / 180
    x = (256/(2*np.pi))*(lng_rad + np.pi)
    y = (256/(2*np.pi))*(np.log(np.tan(np.pi/4 + lat_rad/2)))
    return (1.4*x, 1.25*y)

px, py = lng_lat_to_pixels(max_deaths['Longitude'], max_deaths['Latitude'])
sizes = 0.001*max_deaths['Deaths'].values

# Set up a global map projection
projection = ccrs.PlateCarree()

# Create a figure and axes
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection=projection))

# Add map features to the plot
ax.add_feature(crt.feature.LAND, facecolor='lightgray')
ax.add_feature(crt.feature.OCEAN, facecolor='white')
ax.add_feature(crt.feature.COASTLINE, linewidth=0.5, edgecolor='gray')
ax.add_feature(crt.feature.BORDERS, linewidth=0.5, edgecolor='gray')
ax.add_feature(crt.feature.LAKES, alpha=0.5, facecolor='white')
ax.add_feature(crt.feature.RIVERS, linewidth=0.2, edgecolor='blue')

# Plot the data points
ax.scatter(px-178, py, s=sizes, color='black', transform=projection)

# Set the extent of the plot to the global map
ax.set_extent([-180, 180, -90, 90], crs=projection)

# Save the plot to a png file
plt.savefig('global_map.png', dpi=300)
plt.show()