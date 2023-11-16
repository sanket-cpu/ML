import numpy as np
from sklearn.cluster import KMeans

X = np.array([[5.9, 3.2],[4.6, 2.9],[6.2, 2.8],[4.7, 3.2],[5.5, 4.2],[5.0, 3.0],[4.9, 3.1],[6.7, 3.1],
    [5.1, 3.8],
    [6.0, 3.0],
    # Continue with your 2D data points
])

k = 3

inital_clusters = np.array([[6.2,3.2],[6.6,3.7],[6.5,3.0]])

kmeans = KMeans(n_clusters=k  , init=inital_clusters, n_init=1, max_iter=500 )
kmeans.fit(X)
first_cluster_center = kmeans.cluster_centers_
a = first_cluster_center[0]
print("after one iteration , red is: ", first_cluster_center )
iters = kmeans.n_iter_
print("Number of iterations: ", iters)
