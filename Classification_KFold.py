# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 19:01:55 2022

@author: Ankit Patel
Sapid: 53004200018
"""
# importing all the necessary libraries
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# Reading the Dataset
df = pd.read_csv(C: // Users//ankit//Desktop//MSCIT//DL//DL//DL//Datasets//seeds_dataset.csv')
# split into input (X) and output (y) variables
X = df.iloc[:, 0:7].values
y = df.iloc[:, -1].values
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to one hot encoded
Y = to_categorical(encoded_Y)
# Creating a Neural Network
model = Sequential()
# hidden layer 1 with relu function
model.add(Dense(5, input_dim=7, activation='relu'))
# hidden layer 2 with relu function
model.add(Dense(5, activation='relu'))
# output layer with linear function as it is regression
model.add(Dense(3, activation='softmax'))
# Compiling Neaural Network with Loss function as categorical_crossentropy as its multiclass classification
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
# Using KerasClassifier to estimate the model
estimator = KerasClassifier(model, epochs=100, batch_size=5)
kfold = KFold(n_splits=10, shuffle=True)
# Finding the model accuracy
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Model  Accuracy:", (results.mean()*100))
