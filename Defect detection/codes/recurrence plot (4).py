#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.spatial.distance import pdist, squareform #scipy spatial distance
import sklearn as sk
import sklearn.metrics.pairwise
import matplotlib.pyplot as plt
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LeakyReLU
from keras import metrics
from keras import backend as K
import time
from skimage.transform import resize

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.utils import np_utils

import pandas as pd


# In[87]:


def recurrence_plot(s, eps=None, steps=None):
    if eps==None: eps=0.1
    if steps==None: steps=10000
    d = sk.metrics.pairwise.pairwise_distances(s)
    d = np.floor(d / eps)
    d[d > steps] = steps
    Z = squareform(d)
    return d


# In[106]:


fig = plt.figure(figsize=(15,14))
data = pd.read_csv("D:/LAB Project Data/1D signal stuff/airforce/Data051116_163928_oven/1D_new/defected/point(119,41).csv")
random_series = np.asarray(data)
print (random_series)

plt.plot(random_series)
plt.show()


# In[131]:


fig = plt.figure(figsize=(20,20))
#random_series = np.random.random(1000)
ax = (recurrence_plot(random_series))

#ax = fig.add_subplot(1, 2, 1)
#ax.imshow(recurrence_plot(random_series))
#plt.plot (ax)
plt.imshow(ax)
#ax.imshow(recurrence_plot(random_series))
'''sinus_series = np.sin(np.linspace(0,24,1000))
ax = fig.add_subplot(1, 2, 2)
ax.imshow(recurrence_plot(sinus_series[:,None]));'''


# In[130]:


fig.savefig('D:/LAB Project Data/1D signal stuff/airforce/Data051116_163928_oven/1D_new/fig.jpg')


# In[ ]:




