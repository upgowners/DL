# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 00:05:44 2022

@author: Ankit Patel
Sapid: 53004200018
"""
# importing all the necessary libraries
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
# Reading the Dataset
df = pd.read_csv(
    'C://Users//ankit//Desktop//MSCIT//DL//DL//DL//Datasets//iris.csv')
# split into input (X) and output (y) variables
X = df.iloc[:, 0:4].values
y = df.iloc[:, -1].values
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to one hot encoded
Y = to_categorical(encoded_Y)
# Traing the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=0)
# Creating a Neural Network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(4,)),  # Input Layer
    # Hidden Layer 1 with relu as activation function
    keras.layers.Dense(5, activation=tf.nn.relu),
    # Hidden Layer 2 with relu as activation function
    keras.layers.Dense(5, activation=tf.nn.relu),
    # Output Layer with sigmoid as activation function
    keras.layers.Dense(3, activation=tf.nn.softmax),
])
# Compiling Neaural Network with Loss function as categorical_crossentropy as its multiclass classification
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
# Fitting the model
model.fit(X_train, y_train, epochs=50, batch_size=1)
# Calculating Model Accuracy and Loss
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)
