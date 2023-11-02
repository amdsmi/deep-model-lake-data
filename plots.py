import matplotlib.pyplot as plt
from base_class import BaceClass
import config as cfg
import pandas as pd
import numpy as np
from matplotlib import cm
import math


class Plots(BaceClass):
    def __init__(self,
                 dem_path):
        super().__init__()
        self.initialize()
        self.dem_path = dem_path
        self._data['date'] = self._date

    def make_grid(self):
        data = pd.read_csv(self.dem_path, sep=" ")
        pivot_df = data.pivot(index='y', columns='x', values='z')
        y = pivot_df.index
        x = pivot_df.keys()
        z = pivot_df.values
        x, y = np.meshgrid(x, y)
        return x, y, z

    def variation_plot(self):
        dates = self._data['date'].unique()
        v_min = self._data[cfg.variation_plot_name].min()
        v_max = self._data[cfg.variation_plot_name].max()

        # Loop over the dates and create a subplot for each date
        for i, date in enumerate(dates):
            # Extract the data for the current date
            date_df = self._data[self._data['date'] == date]
            lat = date_df['Lat']
            lon = date_df['Long']
            depth = date_df['Depth']
            target = date_df[cfg.variation_plot_name]
            x, y, z = self.make_grid()
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 6))

            # surf = ax.plot_surface(x, y, z, cmap=cm.terrain,
            #                        linewidth=0, antialiased=False, alpha=0.7, vmax=700)

            sc = ax.scatter(lon, lat, -depth, c=target,vmin=v_min, vmax=v_max,
                            cmap=cfg.variation_color_map)  # reversed colormap

            # Set the axis labels and title
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_zlabel('Depth')
            ax.set_title('{} values in 3D space for {}'.format(cfg.variation_plot_name, str(date)[:10]))

            # Reverse the depth axis tick labels
            ax.set_zlim(ax.get_zlim()[::-1])
            ax.invert_zaxis()

            # Add a colorbar and label
            cbar = plt.colorbar(sc, ax=ax)
            # fig.colorbar(surf, shrink=0.5, aspect=5)
            cbar.ax.set_ylabel(f'{cfg.variation_plot_name} variation')
            plt.savefig(f'plots/variation/{str(date)[:10]}_{cfg.variation_plot_name}.jpg'.format(date), dpi=300, format='jpg')
            # plt.show()


if __name__ == "__main__":
    plot = Plots(cfg.dem_path)
    plot.variation_plot()
