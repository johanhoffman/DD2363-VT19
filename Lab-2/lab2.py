import time
import numpy as np

def backward_substitution(R,b):
    # R is an upper triangular matrix
    x = np.zeros(b.shape)
    n = R.shape[0]
    x[n-1] = b[n-1] / R[n-1,n-1]
    # iterate in reverse order to find x_i = b_i - sum(R_ij*x_j)/R_ii ,for j in [i+1,n)
    for i in range(n-2,-1,-1):
        x[i] = (b[i]-sum([R[i,j]*x[j] for j in range(i+1,n)]))/R[i,i]
    return x

def gs(A):
    #gram-schmidt method for creating a orthonormal matrix Q and upper triangular matrix R from A
    rows,cols = A.shape
    Q = np.zeros((rows,cols))
    R = np.zeros((cols,cols))
    for j in range(cols):
        vj = A[:,j]
        if j > 0: # if j == 0, we don't need to modify the direction of the corresponding vector
            for i in range(j):
                R[i,j] = np.inner(Q[:,i],A[:,j])
                vj = np.subtract(vj,R[i,j]*Q[:,i])
        R[j,j] = np.linalg.norm(vj)
        Q[:,j] = vj/R[j,j]
    return Q, R


def direct_solver(A,b):
    # Ax = b <=> QRx = b <=> Rx = Q^(⁻1)b = Q^(T)b
    (Q,R) = gs(A)
    # use backward backward_substitution to solve this "new" equation system
    return backward_substitution(R,Q.transpose().dot(b))


def least_squares(A,b):
    new_A = np.matmul(np.transpose(A),A)
    # Ax = b is overdetermined
    # normal equations: A^(T)Ax = A^(T)b
    new_b = np.transpose(A).dot(b)
    return direct_solver(new_A, new_b)

# this finds the largest eigenvalue lamda_1 and corresponding eigenvector v_1 for A
def power_iteration(A):
    v_1 = np.random.rand(A.shape[0])
    v_1 *= 1/np.linalg.norm(v_1)
    lamda_1 = 0
    for k in range(100):
        w = A.dot(v_1)
        v_1 = w/np.linalg.norm(w)
        lamda_1 = np.inner(v_1, A.dot(v_1))
    return lamda_1, v_1

if __name__ == '__main__':
    A = np.array([[1,2,3], [2,-2,0], [3,0,4]])
    ev, v = power_iteration(A)
    print ev,v
