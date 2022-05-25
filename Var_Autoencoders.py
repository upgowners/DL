# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:54:36 2022

@author: Ankit Patel
Sapid: 53004200018
"""
# import neccessary libararies
from keras.models import Model
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras import backend as K
from keras.losses import binary_crossentropy
from numpy import reshape
import matplotlib.pyplot as plt
# Loading MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape)
# Shapping the dataset
image_size = x_train.shape[1]
x_train = reshape(x_train, [-1, image_size, image_size, 1])
x_test = reshape(x_test, [-1, image_size, image_size, 1])
print(x_train.shape, x_test.shape)
# trainning the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# Defining the encoder
latent_dim = 8
input_img = Input(shape=(image_size, image_size, 1),)


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


h = Conv2D(16, kernel_size=3, activation='relu',
           padding='same', strides=2)(input_img)
enc_ouput = Conv2D(32, kernel_size=3, activation='relu',
                   padding='same', strides=2)(h)
shape = K.int_shape(enc_ouput)
x = Flatten()(enc_ouput)
x = Dense(16, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder = Model(input_img, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
# Defining the decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)
x = Conv2DTranspose(32, kernel_size=3, activation='relu',
                    strides=2, padding='same')(x)
x = Conv2DTranspose(16, kernel_size=3, activation='relu',
                    strides=2, padding='same')(x)
dec_output = Conv2DTranspose(
    1, kernel_size=3, activation='relu', padding='same')(x)
decoder = Model(latent_inputs, dec_output, name='decoder')
decoder.summary()
# Defining the VAE model
outputs = decoder(encoder(input_img)[2])
vae = Model(input_img, outputs, name='vae')
reconst_loss = binary_crossentropy(K.flatten(input_img), K.flatten(outputs))
reconst_loss *= image_size * image_size
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconst_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()
# fit the model on training data.
vae.fit(x_train, epochs=20, batch_size=128,
        shuffle=True, validation_data=(x_test, None))
# Generating the images
z_mean, _, _ = encoder.predict(x_test)
decoded_imgs = decoder.predict(z_mean)
# visualize the first 10 images of both original and predicted data
n = 10
plt.figure(figsize=(20, 4))
for i in range(10):
    plt.gray()
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
