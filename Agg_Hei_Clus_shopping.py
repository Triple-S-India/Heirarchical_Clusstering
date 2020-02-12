#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Heirarchical Clustering


# In[2]:


#import librabies
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[6]:


#import dataset
customer_data = pd.read_csv('D:DS_TriS/shopping_data.csv')


# In[7]:


customer_data.head()


# In[8]:


customer_data.shape


# In[9]:


#We will retain the Annual Income (in thousands of dollars) and Spending Score (1-100) columns. 
#The Spending Score column signifies how often a person spends money in a mall on a scale of 1 to 100 
#with 100 being the highest spender.
data = customer_data.iloc[:, 3:5].values


# In[10]:


# create the dendrograms for our dataset.
import scipy.cluster.hierarchy as shc      #import scipy for dendograms

plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(data, method='ward'))


# In[11]:


#group the data points into these k(5) clusters


# In[12]:


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)


# In[13]:


plt.figure(figsize=(10, 7))
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')


# In[ ]:




