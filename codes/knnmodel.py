#%%

# basic knn model

from sklearn import datasets
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123)

def euc_dis(instance1, instance2):
    dist = np.sqrt(sum((instance1 - instance2)**2))
    return dist

def knn_classify(X, y, testInstance, k):
    distances = [euc_dis(x, testInstance) for x in X]
    kneighbors = np.argsort(distances)[:k]
    count = Counter(y[kneighbors])
    return count.most_common()[0][0]

predictions = [knn_classify(X_train, y_train, data, 3) for data in X_test]
correct = np.count_nonzero((predictions == y_test) == True)
print(correct/len(X_test)) # accuracy of prediction






#%%
## how to decide the value of k

import matplotlib as plt
import matplotlib.pyplot as pyplt
# %matplotlib inline
from itertools import product
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# generate some random points
n_points = 100
x1 = np.random.multivariate_normal([1, 50], [[1,10],[0,10]], n_points)
x2 = np.random.multivariate_normal([2, 50], [[1,10],[0,10]], n_points)
xsum = np.concatenate([x1, x2])
y = np.array([0] * n_points + [1] * n_points)
print(xsum.shape, y.shape)

# training
classify = []
neighbors = [1, 3, 5, 9, 11, 13, 15, 17, 19]
for i in range(len(neighbors)):
    classify.append(KNeighborsClassifier(n_neighbors = neighbors[i]).fit(xsum, y))

# visualization
x_min, x_max = xsum[:,0].min() - 1, xsum[:,0].max() +1
y_min, y_max = xsum[:,1].min() - 1, xsum[:,1].max() +1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

f, axarr = pyplt.subplots(3,3, sharex = 'col', sharey = 'row', figsize = (15, 12))
for idx, clf, tt in zip(product([0,1,2],[0,1,2]), classify, ['KNN(k=%d)'%k for k in neighbors]):
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, z, alpha = 0.4)
    axarr[idx[0], idx[1]].scatter(xsum[:,0],xsum[:,1], c = y, s = 20)
pyplt.suptitle("k range from 1 to 19")
pyplt.show()
# %%
