import time
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import tri
from matplotlib import axes


def inner_product(x,y):
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        print "error in func<inner_product>: incompatible vectors"
        return
    res = 0
    for i in range(x.size):
        res += x[i] * y[i]
    return res

def matrix_vec_prod(A,x):
    A_shape = A.shape   # matrix A has shape (rows x cols)
    if A_shape[1] != x.size or x.ndim != 1:
        print "error in func<matrix_vec_prod>: incorrect dimensions"
        return
    prod = np.zeros( ( A_shape[0] ) )
    for r in range(A_shape[0]):
        for c in range(A_shape[1]):
            prod[r] += A[r][c]*x[c]
    return prod

def matrix_matrix_prod(A,B):
    if A.shape[1] != B.shape[0]: # if A's cols are not the same in count as B's rows, then AB is undefined
        print "error in func<matrix_matrix_prod>: incompatible matrix dimensions"
        return
    C = np.zeros((A.shape[0],B.shape[1])) # if A is (n x m) and B is (m x p), then C is (n x p)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            for k in range(A.shape[1]):
                C[i][j] += A[i][k]*B[k][j]
    return C


v1 = np.array([1, 2, 3, 4])
A = np.array([1])
B = np.array([2])
print(B.shape[0],B.shape[1],A.shape[0],A.shape[1])
#print(matrix_matrix_prod(B,A))
