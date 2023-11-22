import numpy as np
from sklearn.cluster import KMeans

X = np.array([[5.9, 3.2],[4.6, 2.9],[6.2, 2.8],[4.7, 3.2],[5.5, 4.2],[5.0, 3.0],[4.9, 3.1],[6.7, 3.1],
    [5.1, 3.8],
    [6.0, 3.0],
    # Continue with your 2D data points
])

k = 3

inital_clusters = np.array([[6.2,3.2],[6.6,3.7],[6.5,3.0]])

# First subquestion - run different kmeans with different variables for each question



kmeans = KMeans(n_clusters=k  , init=inital_clusters, n_init=1, max_iter=1 ) # runs only one iteration
kmeans.fit(X)
first_cluster_center = kmeans.cluster_centers_
a = first_cluster_center[0] # the index 0 is red - 1 is green , 2 is blue CONFIRM BASED ON QUESTION
print("after one iteration , red is: ", first_cluster_center )
iters = kmeans.n_iter_
print("Number of iterations: ", iters)

# Second subquestion


km = KMeans(n_clusters=k  , init=inital_clusters, n_init=1, max_iter=2 ) # runs only two iterations 
km.fit(X)
first_cluster_center = km.cluster_centers_
b = first_cluster_center[1] # green COLOR
print("after two iterations , green is: ", b )
iters = km.n_iter_
print("Number of iterations: ", iters) # prints number of iterations

# Third Subquestion 

kmx = KMeans(n_clusters=k  , init=inital_clusters, n_init=1) # runs iterations till convergence
kmx.fit(X)
first_cluster_center = kmx.cluster_centers_
b = first_cluster_center[2] # blue color
print("after cluster converges , blue is: ", b )
iters = kmx.n_iter_
print("Number of iterations: ", iters)


# Fourth subquestion 

boo = KMeans(n_clusters=k  , init=inital_clusters, n_init=1) # runs only iterations till convergence 
boo.fit(X)
first_cluster_center = boo.cluster_centers_

iters = boo.n_iter_
print("Number of iterations for all clusters to converge: ", iters)