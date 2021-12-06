import random
import numpy as np
import numpy.random as nr


def create_temp_sal_data(endpoints, n_points, p_outlier, fuzz=0.1):

    def create_point(endpoint1, endpoint2, r, fuzz):
        return r* (endpoint1+(fuzz*nr.randn(1,2)))+ (1-r) * (endpoint2+(fuzz*nr.randn(1,2)))

    data = np.array([])

    n_valid = int((1-p_outlier) * n_points)
    n_anom = int(p_outlier * n_points)

    for idx in range(n_valid//(len(endpoints)-1)):
        r = random.random()
        points =np.array([create_point(endpoints[j-1], endpoints[j], r, fuzz) for j in range(1,len(endpoints))])
        data = np.append(data, points)
    anomalies = np.ones((n_anom,1))*[32.5, 6]+(fuzz*nr.randn(n_anom,2))

    return data.reshape((len(data)//2, 2)), anomalies
