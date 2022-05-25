# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 20:03:53 2022

@author: Ankit Patel
Sapid: 53004200018
"""
# Importing the neccassary libraries
from tflearn import DNN
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

# Training dataset
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]

# Creating input layer of size 2
input_layer = input_data(shape=[None, 2])
# Creating hidden layer of size 2 with activation function as "tanh"
hidden_layer = fully_connected(input_layer, 2, activation='tanh')
# Creating output layer of size 1 with activation function as "tanh"
output_layer = fully_connected(hidden_layer, 1, activation='tanh')

# Using Stohastic Gradient Descent for optimization and Binary Crossentropy for loss function with a learning rate of 5
regression = regression(output_layer, optimizer='sgd',
                        loss='binary_crossentropy', learning_rate=5)
model = DNN(regression)

# Fitting the model with 5000 iterations
model.fit(X, Y, n_epoch=500, show_metric=True)

# Predicting the values to see the results accuracy
print('Expected:  ', [i[0] > 0 for i in Y])
print('Predicted: ', [i[0] > 0 for i in model.predict(X)])
