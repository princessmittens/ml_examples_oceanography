import mock_data
import plot_data
import models

import numpy as np

n_points = 200
p_outlier = 0.2
n_clusters = 2

# 3 endpoints
endpoints = [[33, 10], [31, 5], [33, 3]]

def main(n_points, p_outlier, n_clusters, endpoints):
   valid_points, anomalies = mock_data.create_temp_sal_data(endpoints, n_points, p_outlier)

   data = np.concatenate((valid_points, anomalies))

   plot_data.plot_data_points(valid_points, anomalies)

   models.run_models(data, n_clusters, plot=True)

if __name__=="__main__":
    main(n_points, p_outlier, n_clusters, endpoints)
