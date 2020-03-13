import pandas as pd
import os
from geopy.distance import geodesic
from gmplot import gmplot
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


stop_line = {
    'south':[42.305160, -83.693049],
    'north':[42.304873, -83.692746],
    'east':[42.304907, -83.693093],
    'west':[42.305136, -83.692679],
}

limit = {
    'lat_min': 42.304088,
    'lat_max': 42.305979,
    'long_min': -83.694080,
    'long_max': -83.691690,
}

direction = {
    'west': 70,
    'east': 255,
    'north':360,
    'south': 185
}

def heading(heading_deg, angle_derivation=20):
    if 70 - angle_derivation < heading_deg < 70 + angle_derivation:  # 90, from west to east
        heading_to = 'east'
    elif 185 - angle_derivation < heading_deg < 185 + angle_derivation:  # 180, from north to south
        heading_to = 'south'
    elif 255 - angle_derivation < heading_deg < 255 + angle_derivation:  # 270, from east to west
        heading_to = 'west'
    elif 360 - angle_derivation < heading_deg < 360 or 0 < heading_deg < angle_derivation:  # 360, from south to north
        heading_to = 'north'
    else:
        heading_to = 'None'
    return heading_to



def plot_on_map(df, out_folder='./', plt_type='heatmap'):
    gmap = gmplot.GoogleMapPlotter(center_lat=df['Latitude'].iloc[0], center_lng=df['Longitude'].iloc[0], zoom=18, apikey='')
    gps_loc = df[['Latitude','Longitude']].values
    gps_lats, gps_lons = zip(*gps_loc)
    if plt_type == 'heatmap':
        gmap.heatmap(gps_lats, gps_lons)
    else:
        gmap.scatter(gps_lats, gps_lons, '#3B0B39', size=1, marker=False)
    out_file = os.path.join(out_folder, 'traffic_heatmap.html')
    gmap.draw(out_file)

input_folder = './Trips'
output_folder = './Processed_data'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

n_create = 10000
show_plot = False
plot_all = False
all_data = pd.DataFrame()
n = 0
mapping = {'Red': 0, 'Green': 1, 'Yellow': 2}
instance_index = 0
plt.figure(0)
plt.xlim(-83.6962, -83.6902)
plt.ylim(42.2979, 42.3081)

for file in os.listdir(input_folder):
    if file.endswith('.csv'):
        file_dir = os.path.join(input_folder, file)
        df = pd.read_csv(file_dir, index_col=False)
        df = df[(df['Latitude'] > limit['lat_min']) & (df['Latitude'] < limit['lat_max'])]
        df = df[(df['Longitude'] > limit['long_min']) & (df['Longitude'] < limit['long_max'])]

        if df.shape[0] < 30:
            print(file[:-4], 'distance too short')
            continue
        heading_angle = df['Heading'].iloc[0:10].mean()
        heading_direction = heading(heading_angle)
        try:
            stop_point = stop_line[heading_direction]
        except KeyError:
            continue
        dist_to_stop = []
        for i in range(df.shape[0]):
            row = df.iloc[i]
            row_loc = [row['Latitude'], row['Longitude']]
            dist = geodesic(stop_point, row_loc).m
            dist_to_stop.append(dist)
        df['Dist'] = dist_to_stop
        df['Instance'] = instance_index
        df = df.replace({'Red': 0, 'Green': 1, 'Yellow': 2})
        df_select = df[['Gentime', 'Latitude', 'Longitude', 'Dist', 'Speed', 'Heading',
                        'Ax', 'Ay', 'Yawrate', 'RadiusOfCurve', 'Status', 'Instance']]
        # plot_on_map(df)

        all_data = pd.concat([all_data,df_select], ignore_index=True)
        instance_index += 1
        n += 1
        if show_plot:
            plt.grid(True)
            plot_x = df_select['Longitude']
            plot_y = df_select['Latitude']
            sns.scatterplot(x=plot_x, y=plot_y)
            # plt.pause(0.001)
            plt.show()
            if n % 1 == 0:
                plt.close()
                plt.figure(n)
        if n >= n_create:
            break

        if instance_index % 500 == 0:
            save_file = os.path.join(output_folder, str(instance_index) + '.csv')
            all_data.to_csv(save_file, index=False)
            all_data = pd.DataFrame()

all_data.to_csv('./last.csv', index=False)

if plot_all:
    y_max, y_min = all_data['Latitude'].max(), all_data['Latitude'].min()
    x_max, x_min = all_data['Longitude'].max(), all_data['Longitude'].min()
    plt.grid(True)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plot_x = all_data['Longitude']
    plot_y = all_data['Latitude']
    sns.scatterplot(x=plot_x, y=plot_y)
    plt.show()