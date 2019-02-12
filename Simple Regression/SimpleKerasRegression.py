
# coding: utf-8

# In[1]:


# Multilayer Perceptron (MLP) for multi-class softmax classification:
# modified from 
# https://keras.io/getting-started/sequential-model-guide/#multilayer-perceptron-mlp-for-multi-class-softmax-classification

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam

import random


# In[2]:


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


# In[3]:


# split data between train and test set 
import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=42)


# In[4]:


import tensorflow as tf

model = Sequential()
n_hidden_units = 256
# Dense(n_hidden_units) is a fully-connected layer with n_hidden_units hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 10-dimensional vectors.
model.add(Dense(n_hidden_units, activation='relu', input_dim=10))
model.add(Dropout(0.5))
model.add(Dense(n_hidden_units, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# optimizer options
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop = tf.train.RMSPropOptimizer(0.008)
adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


# In[5]:


batchsize = 512

model.fit(X_train, y_train,
          epochs=40,
          batch_size=batchsize)
score = model.evaluate(X_test, y_test, batch_size=batchsize)
print(score)

'''
Why is the training loss much higher than the testing loss?

A Keras model has two modes: training and testing. Regularization mechanisms, such as Dropout and 
L1/L2 weight regularization, are turned off at testing time. Besides, the training loss is the average 
of the losses over each batch of training data. Because your model is changing over time, the loss over 
the first batches of an epoch is generally higher than over the last batches. On the other hand, the 
testing loss for an epoch is computed using the model as it is at the end of the epoch, resulting in a lower loss.
'''


# In[6]:


from keras.models import load_model

model.save('MLP_Multiclass_softmax_10_inputs.h5')  # creates a HDF5 file 'my_model.h5'
model.save_weights('MLP_Multiclass_softmax_10_inputs_weights.h5')

'''
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('MLP_Multiclass_softmax_10_inputs.h5')
model.load_weights('MLP_Multiclass_softmax_10_inputs_weights.h5') # for same architecture
model.load_weights('MLP_Multiclass_softmax_10_inputs_weights.h5', by_name=True) # for different architecture
'''

