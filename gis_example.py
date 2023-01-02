import os.path
import subprocess
from typing import Union

import folium
import numpy as np
import pandas as pd

from dbscan import DBSCAN


def main():
    eps = 0.1
    min_points = 10

    data = read_data()
    labels = get_cluster_labels(data, eps, min_points)

    data['Cluster'] = labels
    map_ = create_map(data)
    map_.save('results/gis_example_result.html')


def read_data():
    if not os.path.isfile('data/accidents_2012_to_2014.csv'):
        download_data()

    data = pd.read_csv(
        'data/accidents_2012_to_2014.csv',
        usecols=['Latitude', 'Longitude', 'Number_of_Vehicles', 'Time', 'Local_Authority_(Highway)', 'Year'])

    data = data[data['Year'] == 2014]
    data = data[(data['Local_Authority_(Highway)'] == 'E09000001') | (data['Local_Authority_(Highway)'] == 'E09000033')]
    return data


def download_data():
    os.makedirs('data', exist_ok=True)
    subprocess.run(
        [
            'kaggle', 'datasets', 'download',
            '-d', 'daveianhickey/2000-16-traffic-flow-england-scotland-wales',
            '-p', 'data/'
        ]
    )

    subprocess.run(['unzip', '-q', 'data/2000-16-traffic-flow-england-scotland-wales.zip', '-d', 'data/'])


def get_cluster_labels(data: pd.DataFrame, eps: Union[int, float], min_points: int):
    data_loc = data[['Latitude', 'Longitude']]
    dbc = DBSCAN(
        epsilon=eps,
        min_points=min_points,
        metric=haversine_distance
    ).fit(data_loc.values)

    labels = dbc.labels
    return labels


def haversine_distance(u, v):
    # https://stackoverflow.com/a/4913653/20380842
    lat1, lon1 = u
    lat2, lon2 = v

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    d = 6367 * c

    return d


def create_map(data):
    location = data['Latitude'].mean(), data['Longitude'].mean()

    map_ = folium.Map(location=location, zoom_start=13)
    folium.TileLayer('cartodbpositron').add_to(map_)

    cluster_colours = [
        '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
        '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'
    ]

    for i in range(len(data)):
        colour_i = data['Cluster'].iloc[i]
        if colour_i == -1:
            pass
        else:
            col = cluster_colours[colour_i % len(cluster_colours)]
            folium.CircleMarker(
                (data['Latitude'].iloc[i], data['Longitude'].iloc[i]), radius=10, color=col, fill=col
            ).add_to(map_)

    return map_


if __name__ == '__main__':
    main()
