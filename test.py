import pandas as pd
import matplotlib.pyplot as plt

# Read the data into a pandas dataframe
df = pd.read_excel('/content/drive/MyDrive/Water quality paper/Data/Wadi Dayqah Dam data- 8sets Variation.xlsx')

# Get the unique dates in the dataframe
dates = df['Date'].unique()

# Create a figure with subplots for each date
nrows = 5
ncols = 2
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 6*nrows), subplot_kw={'projection': '3d'})

# Loop over the dates and create a subplot for each date
for i, date in enumerate(dates):
    # Extract the data for the current date
    date_df = df[df['Date'] == date]
    lat = date_df['Lat']
    lon = date_df['Long']
    depth = date_df['Depth [m]']
    do = date_df['DO [mg/l]']

    # Plot the data for the current date in a subplot
    ax = axs[i]
    sc = ax.scatter(lon, lat, -depth, c=do, vmin=do.min(), vmax=do.max(), cmap='coolwarm_r') # reversed colormap

    # Set the axis labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Depth')
    ax.set_title('DO values in 3D space for {}'.format(date))

    # Reverse the depth axis tick labels
    ax.set_zlim(ax.get_zlim()[::-1])
    ax.invert_zaxis()

    # Add a colorbar and label
    cbar = plt.colorbar(sc, ax=ax)
    cbar.ax.set_ylabel('DO variation')

# Save the plot to a file
plt.savefig('Do_all.jpg', dpi=300, format='jpg')

# Show the plot
plt.show()