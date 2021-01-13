#!/usr/bin/env python
# coding: utf-8

# Decision trees: fruit classification

# In[12]:


from sklearn import tree


# In[13]:


# 0 and 1 represents bumpy and smooth, and apple and orange in features and labels respectively
features = [[140,1],[130,1], [150,0], [170, 0]]
labels = [0, 0, 1, 1]


# In[10]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)


# In[14]:


print(clf.predict([[150,0]]))


# In[15]:


# The classifier predicts an orange


# Decision trees: iris classification

# In[ ]:





# In[30]:


import numpy as np
from sklearn.datasets import load_iris


# In[18]:


iris = load_iris()


# In[20]:


print(iris.feature_names)
print(iris.target_names)


# In[21]:


print(iris.data[0])


# In[22]:


print(iris.target[0])


# In[26]:


for i in range(len(iris.target)):
    print("Exanple %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))


# In[45]:


print(iris)


# In[31]:


test_idx = [0, 50, 100]


# In[33]:


# training data: remove the above the data and target variables from three entries for testing 


# In[34]:


train_target = np.delete(iris.target, test_idx)


# In[35]:


train_data = np.delete(iris.data, test_idx, axis=0)


# In[36]:


# prepare test data


# In[37]:


test_target = iris.target[test_idx]
test_data = iris.data[test_idx]


# In[38]:


clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)


# In[40]:


print(test_target)


# In[41]:


print(clf.predict(test_data))


# In[42]:


test_data


# In[ ]:





# In[ ]:





# In[ ]:




