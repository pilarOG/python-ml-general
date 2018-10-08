# https://www.udemy.com/mathematical-foundation-for-machine-learning-and-ai/learn/v4/t/lecture/10496544?start=0
# Norms and Eigendecomposition

import numpy as np
from numpy import linalg

# Vector
A = np.arange(9) - 3
print A

# Matrix
B = A.reshape((3,3))
print B

# Euclidean L2 norm

print np.linalg.norm(A)
print np.linalg.norm(B)

# Frogenius norm is the L2 norm for a matrix
print np.linalg.norm(B, 'fro')

# max norm (P = infinity), not the same for vector and matrix
print np.linalg.norm(A, np.inf)
print np.linalg.norm(B, np.inf)

# vector normalization - normalization to produce a unit vector
# essential preprocessing step
norm = np.linalg.norm(A)
A_unit = A / norm
print A_unit

# the magnitude of a unit vector is equal to 1
print np.linalg.norm(A_unit)

# eigendecomposition

# find the eigenvalues and eigenvectors for a simple squeare matrix
# diagonal matrix
A = np.diag(np.arange(1,4))
print A

eigenvalues, eigenvectors = np.linalg.eig(A)
# each column is an eigenvector
print eigenvectors
# and its corresponding eigenvalues w[i] to the eigenvector v[:,i]
print eigenvalues

# verify if decomposition is correct, the equation should give the original matrix A
matrix = np.matmul(np.diag(eigenvalues), np.linalg.inv(eigenvectors))
output = np.matmul(eigenvectors, matrix).astype(np.int)
print output
