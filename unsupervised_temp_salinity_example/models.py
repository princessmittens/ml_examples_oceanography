from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering

import plot_data

def run_models(data, n_clusters, plot=True):
    results = {}

    kmeans = KMeans(n_clusters=n_clusters).fit(data)
    results["K-Means, 2 cluster"] = kmeans.predict(data)

    agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage="single").fit(data)
    results["Agglomerative Clustering"] = agglo.labels_

    spectral = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', affinity="nearest_neighbors").fit(data)
    results['Spectral'] = spectral.labels_

    dbscan = DBSCAN().fit(data)
    results['DBSCAN'] = dbscan.labels_

    gm = GaussianMixture(n_components=2).fit(data)
    results['Gaussian Mixture'] = gm.predict(data)

    if plot:
        for k,v in results.items():
            plot_data.plot_results(k, data, v)
