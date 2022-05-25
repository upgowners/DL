# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 00:42:04 2022

@author: Ankit Patel
Sapid: 53004200018
"""
# Import neccessary libaries
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D
# scale and make train and test dataset
(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.astype('float32')/255.
X_test = X_test.astype('float32')/255.
X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))
print(X_train.shape)
print(X_test.shape)
# Adding Noisy data
noise_factor = 0.5
X_train_noisy = X_train + noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
# to make values in the range of 0 to 1,
# if values < 0 then they will be equal to 0 and
# if values > 1 then they will be equal to 1.
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)
# Taking the input image
input_img = Input(shape=(28, 28, 1))
# encoding architecture
x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x1 = MaxPool2D((2, 2), padding='same')(x1)
x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
x2 = MaxPool2D((2, 2), padding='same')(x2)
x3 = Conv2D(16, (3, 3), activation='relu', padding='same')(x2)
encoded = MaxPool2D((2, 2), padding='same')(x3)
# decoding architecture
x3 = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x3 = UpSampling2D((2, 2))(x3)
x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x3)
x2 = UpSampling2D((2, 2))(x2)
x1 = Conv2D(64, (3, 3), activation='relu')(x2)
x1 = UpSampling2D((2, 2))(x1)
decoded = Conv2D(1, (3, 3), padding='same')(x1)
# creating autoencoder model
autoencoder = keras.Model(input_img, decoded)
# Compiling the autoencoder model with binary crossentropy loss, and the Adam optimizer
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# train autoencoder with training dataset with 20 epochs and using minibatch approach
autoencoder.fit(X_train_noisy, X_train, epochs=20, batch_size=256,
                shuffle=True, validation_data=(X_test_noisy, X_test))
# to predict the reconstructed images for the original images...
pred = autoencoder.predict(X_test_noisy)
plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.xticks([])  # to remove x-axis  the [] empty list indicates this
    plt.yticks([])  # to remove y-axis
    plt.grid(False)  # to remove grid
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')  # display the image
plt.tight_layout()  # to have a proper space in the subplots
plt.show()

plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.xticks([])  # to remove x-axis  the [] empty list indicates this
    plt.yticks([])  # to remove y-axis
    plt.grid(False)  # to remove grid
    plt.imshow(X_test_noisy[i].reshape(28, 28),
               cmap='gray')  # display the image
plt.tight_layout()  # to have a proper space in the subplots
plt.show()

# to visualize reconstructed images(output of autoencoder)
plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.xticks([])  # to remove x-axis  the [] empty list indicates this
    plt.yticks([])  # to remove y-axis
    plt.grid(False)  # to remove grid
    plt.imshow(pred[i].reshape(28, 28), cmap='gray')  # display the image
plt.tight_layout()  # to have a proper space in the subplots
plt.show()
