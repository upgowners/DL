# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 01:07:08 2022

@author: Ankit Patel
Sapid: 53004200018
"""
# import all the  necessary Libraries
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# Reading the Dataset
df = pd.read_csv(
    'C://Users//ankit//Desktop//MSCIT//DL//DL//DL//Datasets//Real estate.csv')
# split into input (X) and output (y) variables
X = df.iloc[:, 2:7].values
Y = df.iloc[:, -1].values
# define wider model
model = Sequential()
model.add(Dense(6, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='linear'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=50, verbose=1)
# Calculating Model Accuracy and Loss
test_loss = model.evaluate(X, Y, verbose=0)
print('Test Loss:', test_loss)
