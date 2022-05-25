# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 23:08:18 2022

@author: Ankit Patel
Sapid: 53004200018
"""

# import all the  necessary Libraries
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
# Reading the Dataset
df = pd.read_csv(
    'C://Users//ankit//Desktop//MSCIT//DL//DL//DL//Datasets//Real estate.csv')
# split into input (X) and output (y) variables
X = df.iloc[:, 2:7].values
Y = df.iloc[:, -1].values
# define wider model
model = Sequential()
# hidden layer 1 with relu function
model.add(Dense(6, input_dim=5, kernel_initializer='normal', activation='relu'))
# hidden layer 2 with relu function
model.add(Dense(6, activation='relu'))
# output layer with linear function as it is regression
model.add(Dense(1, activation='linear'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')
# evaulating the model using KFold method
estimator = KerasRegressor(model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, Y, cv=kfold)
# Calculating Model Mean loss and standard deviation
print("Model: %.2f (%.2f) MSE" % (results.mean(), results.std()))
