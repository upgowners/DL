# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 17:35:26 2022

@author: Ankit Patel
Sapid 53004200018
"""
import numpy as np
# Creating Vector
x = np.array([1, 2, 3, 4])
print("1 Diemsonal Vector", x)
A = np.array([[1, 2], [3, 4], [5, 6]])
print("Matrix A", A)

# Transpose of Matrix A
A_t = A.T
print("Transpose of Matrix A", A_t)

# Creating Matrix B
B = np.array([[2, 5], [7, 4], [4, 3]])
print("Matrix B", B)

# Add matrices A and B
C = A + B
print("Addition of Matrix A and B", C)

# Exemple: Add 4 to the matrix A
C = A+4
print("Addition of a scalar value to Matrix C", C)

# Creating a matrix of diffrent shape
D = np.array([[2], [4], [6]])
print("Matrix D", D)

# Broadcasting
C = A+D
print("Broadcasting", C)

# Using Dot Function
B = np.array([[2], [4]])
C = np.dot(A, B)
print("Dot function using Numpy", C)

C = A.dot(B)
print("Dot function without using Numpy", C)

# Distributive Matrix
C = np.array([[4], [3]])
D = A.dot(B+C)
print("Distributive Matrix", D)

# Associative Matrix
B = np.array([[5, 3], [2, 2]])
D = A.dot(B.dot(C))
print("Associative Matrix", D)

# Inverse Matrix
A = np.array([[3, 0, 2], [2, 0, -2], [0, 1, 1]])
A_inv = np.linalg.inv(A)
print("Inverse of Matrix A", A_inv)

A_bis = A_inv.dot(A)
print("Inverse using dot function", A_bis)
