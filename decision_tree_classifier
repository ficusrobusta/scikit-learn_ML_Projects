#Task 1: Decision trees: fruit classification

from sklearn import tree

# 0 and 1 represents bumpy and smooth, and apple and orange in features and labels respectively
features = [[140,1],[130,1], [150,0], [170, 0]]
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print(clf.predict([[150,0]]))

# The classifier predicts an orange

Decision trees: iris classification



import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

print(iris.feature_names)
print(iris.target_names)

print(iris.data[0])

print(iris.target[0])

for i in range(len(iris.target)):
    print("Exanple %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))

print(iris)

test_idx = [0, 50, 100]

# training data: remove the above the data and target variables from three entries for testing 

train_target = np.delete(iris.target, test_idx)

train_data = np.delete(iris.data, test_idx, axis=0)

# prepare test data

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)

print(clf.predict(test_data))

test_data

fig, ax = plt.subplots(figsize=(10, 10)) 
tree.plot_tree((clf), ax=ax)
plt.figure(figsize=(14,14))
plt.show()

Decision trees: dog classification

import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height,lab_height], stacked=False, color=['r','b'])
plt.show()
