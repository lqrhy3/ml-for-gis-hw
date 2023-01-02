from typing import List, Dict, Callable, Set

import numpy as np
from scipy.spatial.distance import euclidean


class DBSCAN:
    def __init__(
            self,
            epsilon: int or float,
            min_points: int,
            metric: Callable = euclidean
    ):
        assert epsilon >= 0.
        assert min_points >= 1

        self.epsilon = epsilon
        self.min_points = min_points
        self.metric = metric

        self._visited_points = None
        self._neighbours = None
        self._clusters = None
        self.labels = None

    def fit_predict(self, x: np.ndarray):
        self.fit(x)
        return self.labels

    def fit(self, x: np.ndarray):
        if x.ndim == 1:
            x = x[:, None]
        assert x.ndim == 2

        num_points = x.shape[0]
        self._visited_points: Set[int] = set()
        self._neighbours: Dict[int, List[int]] = {}
        self._clusters: List[List[int]] = []

        for point_idx in range(num_points):
            if self._is_visited(point_idx):
                continue

            self._neighbours[point_idx] = self._get_neighbors(x, point_idx)
            if self._is_core(point_idx):
                self._visited_points.add(point_idx)
                cluster = self._expand_cluster(x, point_idx)
                self._clusters.append(cluster)

        self.labels = self._get_cluster_labels(x)
        return self

    def _is_visited(self, point_idx: int):
        return point_idx in self._visited_points

    def _get_neighbors(self, x: np.ndarray, point_idx: int):
        neighbors = []
        for idx in range(len(x)):
            if idx == point_idx:
                continue
            distance = self.metric(x[idx], x[point_idx])
            if distance < self.epsilon:
                neighbors.append(idx)

        return neighbors

    def _is_core(self, point_idx: int):
        return len(self._neighbours[point_idx]) + 1 >= self.min_points

    def _expand_cluster(self, x: np.ndarray, point_idx: int):
        cluster = [point_idx]

        for neighbor_idx in self._neighbours[point_idx]:
            if not self._is_visited(neighbor_idx):
                self._visited_points.add(neighbor_idx)
                self._neighbours[neighbor_idx] = self._get_neighbors(x, neighbor_idx)
                if self._is_core(neighbor_idx):
                    expanded_cluster = self._expand_cluster(x, neighbor_idx)
                    cluster.extend(expanded_cluster)
                else:
                    cluster.append(point_idx)

        return cluster

    def _get_cluster_labels(self, x: np.ndarray):
        labels = np.full(len(x), fill_value=-1)
        for cluster_idx, cluster in enumerate(self._clusters):
            labels[cluster] = cluster_idx

        return labels

