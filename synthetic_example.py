from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

from dbscan import DBSCAN


def main():
    X, y = datasets.make_moons(n_samples=500, noise=0.05)
    X = StandardScaler().fit_transform(X)

    dbscan = DBSCAN(epsilon=0.3, min_points=7)
    labels = dbscan.fit_predict(X)
    plot(X, labels)


def plot(X, labels):
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    legend = [f'cluster_{i}' for i in range(1, num_clusters + 1)] + ['noise'] * (-1 in labels)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.legend(handles=scatter.legend_elements()[0], labels=legend)

    plt.savefig('result/synthetic_example_result.png')


if __name__ == '__main__':
    main()
