# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:05:43 2022

@author: 91704
"""
import keras.backend as K
from keras.layers import Dense, Input, Lambda
from keras.models import Model

input_shape = (10,)
intermediate_dim = 10
latent_dim = 5

inputs = Input(shape = input_shape, name = 'encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
mu = Dense(latent_dim, name='mu')(x)
log_var = Dense(latent_dim, name='log_sigma')(x)

def sample(a):
    mu, sigma = a
    epsilon = K.random_normal(shape=(1, latent_dim))
    return mu + K.exp(.5*log_var) * epsilon

z = Lambda(sample, output_shape=(latent_dim,), name='z')([mu, log_var])

encoder = Model(inputs, z, name='encoder')
encoder.summary()

latent_inputs = Input(shape = (latent_dim,), name = 'decoder_input')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(input_shape[0], activation='sigmoid')(x)

decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

outputs = decoder(encoder(inputs))
vae = Model(inputs, outputs, name='VAE')

def vae_loss(input_vols, output_vols):
    beta = 1e-7
    kl_loss = K.sum(-1 - K.log(K.exp(log_var)) + K.exp(log_var) + K.square(mu))/2
    return K.mean((input_vols-output_vols)**2) + beta*kl_loss

vae.compile(optimizer='rmsprop', loss=vae_loss)

vae.summary()

