#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers, datasets, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


(x_train,y_train),(x_test,y_test) = datasets.cifar10.load_data()


# In[3]:


x_train.shape
y_train.shape


# In[4]:


x_test.shape


# In[5]:


x_train


# In[6]:


class_names = ['airplane','automobile', 'bird', 'cat', 'deer', 'dog','frog','horse','ship', 'truck']


# In[7]:


y_train=y_train.reshape(-1)
y_train


# In[8]:


def plot_sample(x,y,index):
    plt.figure(figsize = (15,3))
    plt.imshow(x[index])
    plt.xlabel(class_names[y[index]])
    



# In[9]:


plot_sample(x_train,y_train, 4)


# In[10]:


x_train = x_train/255
x_test = x_test/255


# In[11]:


# model = models.Sequential([
#     layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),
#     layers.MaxPooling2D((2,2)),
    
#     layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
#     layers.MaxPooling2D((2,2)),
    
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])
model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.BatchNormalization(),
    
    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.BatchNormalization(),
    
    layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.BatchNormalization(),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Dropout with 50% rate
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Dropout with 50% rate
    layers.Dense(10, activation='softmax')
])


# In[12]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[13]:


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


# In[14]:


# Define a learning rate scheduler
lr_scheduler = ReduceLROnPlateau(
    monitor='loss',  # Monitor validation loss
    factor=0.1,  # Reduce learning rate by a factor of 10
    patience=5,  # Wait for 5 epochs to see no improvement
    verbose=1  # Print a message when reducing the learning rate
)


# In[15]:


model.fit(x_train,y_train, epochs=50, callbacks=[lr_scheduler])


# In[16]:


model.evaluate(x_test,y_test)


# In[17]:


y_test = y_test.reshape(-1)
y_test


# In[18]:


plot_sample(x_test,y_test,9)


# In[19]:


y_pred = model.predict(x_test)


# In[20]:


y_pred[9]


# In[21]:


np.argmax(y_pred[9])


# In[22]:


class_names[1]


# In[23]:


y_class = [np.argmax(item) for item in y_pred]
y_class[9]


# In[24]:


class_names[y_class[9]]


# In[25]:


import cv2


# In[70]:


img = cv2.imread('ship1.jpeg')


# In[71]:


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(img_rgb)
plt.show()


# In[72]:


# Resize to match model input (assuming 32x32)
img_rgb = cv2.resize(img_rgb, (32, 32))

# Normalize pixel values (assuming scale of 0 to 1)
img_rgb = img_rgb.astype('float32') / 255.0


# In[73]:


img_rgb = np.expand_dims(img_rgb, axis=0)


# In[74]:


# Make prediction using your trained model
img_pred = model.predict(img_rgb)


# In[75]:


np.argmax(img_pred)


# In[76]:


class_names[8]

