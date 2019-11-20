#%%
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


# %%
