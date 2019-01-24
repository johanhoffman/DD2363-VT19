import time
import numpy as np

def backward_substitution(R,b):
    x = np.zeros(b.shape)
    n = R.shape[0]
    x[n-1] = b[n-1] / R[n-1,n-1]
    for i in range(n-2,-1,-1):
        print i
        x[i] = (b[i]-sum([R[i,j]*x[j] for j in range(i+1,n)]))/R[i,i]
    return x


def qr_factorization(A):
    Q = np.zeros( (A.shape) )
    R = np.zeros( (A.shape) )
    for j in range(A.shape[1]):
        Q[:,j] = A[:,j]
        for k in range(j):
            Q[:,j] -= (Q[:,k].dot(A[:,j]))*Q[:,k]
            R[k,j] = (Q[:,k].dot(A[:,j]))
        R[j,j] = np.linalg.norm(Q[:,j])
        Q[:,j] *= 1/np.linalg.norm(Q[:,j])
    return Q,R

def direct_solver(A,b):
    (Q,R) = qr_factorization(A)
    return backward_substitution(R,Q.transpose().dot(b))

A = np.array([[1,2,3], [-1,3,4], [0,2,2]])
y = np.array([1,2,3])
print direct_solver(A,y)
