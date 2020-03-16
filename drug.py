#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt


# In[3]:


my_data=pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv')
my_data.head()


# In[4]:


X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]


# In[5]:


le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]


# In[6]:


y = my_data["Drug"]
y[0:5]


# In[7]:


X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)


# In[8]:


drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree


# In[9]:


drugTree.fit(X_trainset,y_trainset)


# In[10]:


predTree = drugTree.predict(X_testset)
print (predTree [0:5])
print (y_testset [0:5])


# In[11]:


print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# In[13]:


#visualization on its way


# In[ ]:




