#!/usr/bin/env python
# coding: utf-8

# # Covariance and Correlation

# Covariance :

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[5]:


x=pd.Series([12,25,68,42,113])
y=pd.Series([11,29,58,121,100])


# In[7]:


df=pd.DataFrame()


# In[10]:


df["X"]=x
df["Y"]=y


# In[11]:


df


# In[36]:


fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15,3))

ax0.scatter(df["X"], df["X"])
ax1.scatter(df["X"], df["Y"])
ax2.scatter(df["X"]*2, df["Y"]*2)

ax0.set_title("Covariance "+str(np.cov(df["X"], df["X"])[0,1]))
ax1.set_title("Covariance "+str(np.cov(df["X"], df["Y"])[0,1]))
ax2.set_title("Covariance "+str(np.cov(df["X"]*2, df["Y"]*2)[0,1]))


# In[54]:


df["a"]=df["X"]*2
df["b"]=df["Y"]*2


# In[55]:


df


# In[58]:


fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15,3))

ax0.scatter(df["X"], df["X"])
ax1.scatter(df["X"], df["Y"])
ax2.scatter(df["a"], df["b"])

ax0.set_title("Correlation "+str(df["X"].corr(df["X"])))
ax1.set_title("Correlation "+str(df["X"].corr(df["Y"])))
ax2.set_title("Correlation "+str(df["a"].corr(df["b"])))


# In[ ]:




