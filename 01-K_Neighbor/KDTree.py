import numpy as np
from sklearn.neighbors import KDTree 
np.random.seed(0)
X = np.random.random((10, 3))

tree = KDTree(X, leaf_size = 2)

dist, ind = tree.query(X[:1], k=2)
print(ind)
print(dist)
