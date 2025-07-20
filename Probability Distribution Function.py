#!/usr/bin/env python
# coding: utf-8

# # Probability Mass Function

# In[25]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[3]:


import random


# Pmf of a dice thrown

# In[4]:


l=[]
for i in range(10000):
    l.append(random.randint(1,6))


# In[5]:


l[:5]


# In[6]:


s=(pd.Series(l).value_counts()/pd.Series(l).value_counts().sum()).sort_index()
s


# In[7]:


s.plot(kind="bar")


# pmf of two dice thrown

# In[8]:


l=[]
for i in range(10000):
    a=random.randint(1,6)
    b=random.randint(1,6)
    
    l.append(a+b)


# In[9]:


s=(pd.Series(l).value_counts()/pd.Series(l).value_counts().sum()).sort_index()
s


# In[10]:


s.plot(kind="bar")


# # Parametric Density Estimation

# In[11]:


from numpy.random import normal


# In[12]:


sample=normal(loc=50, scale=5, size=1000)
sample


# In[13]:


sample_mean=sample.mean()
sample_mean


# In[14]:


sample_std=sample.std()
sample_std


# In[15]:


plt.hist(sample, bins=10)


# In[16]:


from scipy.stats import norm


# In[17]:


dist=norm(sample_mean, sample_std)


# In[18]:


values=np.linspace(sample.min(), sample.max(), 100)


# In[19]:


probabilities=[dist.pdf(values) for values in values]


# In[20]:


plt.hist(sample, bins=10, density=True)
plt.plot(values, probabilities)


# In[21]:


import seaborn as sns


# In[23]:


sns.distplot(sample)


# # Kernel Density for Non Parametric Density Estimation

# In[49]:


sample1=normal(loc=20, scale=5, size=300)
sample2=normal(loc=40, scale=5, size=700)
sample=np.hstack((sample1, sample2))
sample


# In[50]:


plt.hist(sample, bins=50)


# In[52]:


from sklearn.neighbors import KernelDensity


# In[84]:


kde=KernelDensity(bandwidth=1, kernel="gaussian")


# In[85]:


sample=sample.reshape(len(sample), 1)


# In[86]:


kde.fit(sample)


# In[87]:


values=np.linspace(sample.min(), sample.max(), 100)


# In[88]:


values=values.reshape(len(values), 1)


# In[89]:


probabilities=kde.score_samples(values)


# In[90]:


probabilities=np.exp(probabilities)


# In[91]:


plt.hist(sample, bins=50, density=True)
plt.plot(values, probabilities)
plt.show()


# In[105]:


sns.kdeplot(sample, bw_adjust=1)


# In[ ]:




