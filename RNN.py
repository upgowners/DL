# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 23:31:10 2022
@author: Ankit Patel
Sapid: 53004200018
"""
# Import all the neccessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
# Reading the dataset
df = pd.read_csv(C: // Users//ankit//Desktop//MSCIT//DL//DL//DL//Datasets//TESLA.csv')
df.shape
df = df['Open'].values
df = df.reshape(-1, 1)
# Split the data into training and testing sets
dataset_train = np.array(df[:int(df.shape[0]*0.8)])
dataset_test = np.array(df[int(df.shape[0]*0.8):])
# Scaling our data between zero and one using MinMaxScaler.
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_test = scaler.transform(dataset_test)
# Function for creating datasets


def create_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x, y


# Creating our training and testing data
x_train, y_train = create_dataset(dataset_train)
x_test, y_test = create_dataset(dataset_test)
# Reshaping our data to make it a 3D array in order to use it in LSTM Layer.
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# Building Model
model = Sequential()
model.add(LSTM(units=96, return_sequences=True,
          input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=96, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))
model.add(Dropout(0.2))
model.add(Dense(units=1))
# Compiling the model
model.compile(loss='mean_squared_error', optimizer='adam')
# Fitting the model
model.fit(x_train, y_train, epochs=50, batch_size=32)
# Getting the data ready for actucal and predicted output
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
# Data Visualization
plt.figure(figsize=(10, 8))
plt.plot(y_test_scaled, color='black', label='Tesla Stock Price')
plt.plot(predictions, color='red', label='Predicted Tesla Stock Price')
plt.legend()
