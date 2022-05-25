# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 23:16:43 2022

@author: Ankit Patel
Sapid: 53004200018
"""
# Importing Tensorflow
import tensorflow as tf

# Creating Matrix A
e_matrix_A = tf.random.uniform(
    [2, 2], minval=3, maxval=10, dtype=tf.float32, name="matrixA")
print("Matrix A: \n{}\n\n".format(e_matrix_A))

# Calculating the eigen values and vectors using tf.linalg.eigh function of tensorflow
eigen_values_A, eigen_vectors_A = tf.linalg.eigh(e_matrix_A)
print("Eigen Vectors: \n{} \n\nEigen Values: \n{}\n".format(
    eigen_vectors_A, eigen_values_A))

# Multiplying our eigen vector by random number
sv = tf.multiply(5, eigen_vectors_A)
print(sv)
