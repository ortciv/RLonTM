
# coding: utf-8

# In[71]:


import keras as k
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import random


# In[82]:


import numpy as np

n_samples = 1000000
n_partitions = 10

# Generate tasks and partitions
partition_data = np.random.random((n_samples, n_partitions)) # partition data generation

data = np.zeros((n_samples, n_partitions))                   # initialize input layer
labels = np.zeros((n_samples, n_partitions))                 # initialize outputs layer for training 

task_data = np.zeros((n_samples, 1))                         # initialize task list

for i in range (0, n_samples):
    
    partitions = partition_data[i]
    task = random.uniform(0, partitions.max())
    task_data[i] = task
    
    best_partition = -1
    best_fit = 999999999
    
    for j in range (0, n_partitions):
        current_fit = partitions[j] - task
        data[i,j] = current_fit
        if current_fit > 0 and current_fit < best_fit:
            best_fit = current_fit
            best_partition = j
    
    labels[i][best_partition] = 1


# In[73]:


print(data)
print(labels)


# In[86]:


import tensorflow as tf

# Construct neural network
n_nodes = 32

model = Sequential()
model.add(Dense(n_nodes, input_dim=n_partitions, activation='relu'))
model.add(Dense(n_nodes, activation='relu'))
model.add(Dense(n_partitions, activation='softmax'))

# Configure a model for categorical classification. from https://www.tensorflow.org/guide/keras#train_and_evaluate
model.compile(optimizer=tf.train.RMSPropOptimizer(0.008),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])


# In[87]:


model.fit(data, labels, epochs=100, batch_size=256)

