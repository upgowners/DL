# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 23:32:03 2022

@author: Ankit Patel
Sapid: 53004200018
"""
# importing all the necessary libraries
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Reading the Dataset
df = pd.read_csv(
    'C://Users//ankit//Desktop//MSCIT//DL//DL//DL//Datasets//molecular_activity.csv')
# split into input (X) and output (y) variables
X = df.iloc[:, 0:4].values
y = df['Activity']
# Traing the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
# Creating a Neural Network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(4,)),  # Input Layer
    # Hidden Layer 1 with relu as activation function
    keras.layers.Dense(10, activation=tf.nn.relu),
    # Hidden Layer 2 with relu as activation function
    keras.layers.Dense(10, activation=tf.nn.relu),
    # Output Layer with sigmoid as activation function
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])
# Compiling Neaural Network with Loss function as Cross-entropy
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
# Fitting the model
model.fit(X_train, y_train, epochs=50, batch_size=1)
# Calculating Model Accuracy and Loss
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)
