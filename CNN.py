# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 01:47:01 2022

@author: Ankit Patel
Sapid: 53004200018
"""
# importing all the necessary libraries
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt
# download mnist data and split into train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# plot the first image in the dataset
plt.imshow(X_train[0])
plt.show()
print(X_train[0].shape)
# reshape dataset to have a single channel
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
# one hot encode target values
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
Y_train[0]
print(Y_train[0])
# Creating a CNN Model
model = Sequential()
# Creating Convolutional Layer with relu function
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
# Adding MaxPooling Layer of 2*2
model.add(MaxPooling2D((2, 2)))
# Flattening layer which acts as input to fully connected network
model.add(Flatten())
# Fully connected network with relu function
model.add(Dense(100, activation='relu'))
# Output Layer with softmax function
model.add(Dense(10, activation='softmax'))
# Compiling Model with Loss function as categorical_crossentropy as its multiclass classification
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
# fitting the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=3)
# Evaluate the results
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
