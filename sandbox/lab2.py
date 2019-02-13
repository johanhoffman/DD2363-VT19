import numpy as np

def normalize(v):
    return v / np.linalg.norm(v)

def norm(v):
    return np.linalg.norm(v)

# def gram_schmidt(A):
    # Q = np.zeros_like(A, dtype = np.float64)
    # Q[:,0] = normalize(A[:,0])
    # for j in range(1, A.shape[1]):
        # col = A[:,j].copy()
        # for i in range(j):
            # col -= np.dot(A[:,j], Q[:,i]) * Q[:,i]
        # Q[:,j] = normalize(col)
    # return Q

def factorize_qr(A):
    n = A.shape[0]
    V = A.copy()
    Q = np.zeros_like(A, dtype = np.float64)
    R = Q.copy()
    for i in range(A.shape[0]):
        r_ii = norm(V[:,i])
        R[i,i] = r_ii
        q_i = V[:,i] / r_ii
        for j in range(i+1, n):
            r_ij = np.dot(q_i, V[:,j])
            V[:,j] -= r_ij * q_i
            R[i,j] = r_ij
        Q[:,i] = q_i
    return Q, R

def solve_QR(A, b):
    n = A.shape[0]
    Q, R = factorize_qr(A)
    b_q = np.dot(Q.T, b)
    x = np.zeros_like(b)
    for j in range(n-1, -1, -1):
        x_sum = np.dot(R[j, j+1:], x[j+1:])
        x[j] = (b_q[j] - x_sum) / R[j,j]
    return x

def solve_least_squares(A, b):
    new_A = np.dot(A.T, A)
    new_b = np.dot(A.T, b)
    return solve_QR(new_A, new_b)

def is_diagonal(A):
    return np.allclose(A, np.diag(np.diag(A)))

def is_upper_triangular(A):
    n = A.shape[0]
    for row in range(n):
        if not np.isclose(A[row,:row].sum(), 0, atol = 1e-15):
            return False
    return True

def get_eigenvalues(A):
    A_c = A.copy()
    n = A.shape[0]
    pQ = np.eye(n)
    while not is_upper_triangular(A_c):
    # while not is_diagonal(A_c):
        Q, R = factorize_qr(A_c)
        pQ = np.dot(pQ, Q)
        A_c = np.dot(R, Q)
    eigs = np.diag(A_c)
    return eigs, pQ


def test_QR_decomposition():
    n = 5
    A = np.random.rand(n, n)
    Q, R = factorize_qr(A)
    # check that R is upper triangular
    for row in range(n):
        assert np.isclose(R[row,:row].sum(), 0)
    # assert that Q is orthogonal
    assert np.allclose(np.dot(Q.T, Q).sum(axis = 1), 1)
    # assert that A = Q*R
    assert np.allclose(A, np.dot(Q, R))
    # assert that Q * Q.T is the identity matrix
    assert np.allclose(np.eye(n), np.dot(Q, Q.T))

def test_QR_solve():
    A = np.array([[1, 2], [3, 4]], dtype = np.float64)
    x = np.array([5, 6])
    b = np.dot(A, x)
    x_test = solve_QR(A, b)
    assert np.allclose(x, x_test)
    # print("true_value:", np.dot(A, x))
    # print("test_x:", x_test)
    # print("test_value:", np.dot(A, x_test))
    
def test_least_squares():
    A = np.array([[1, -1], [1, 1], [2, 1]], dtype = np.float64)
    b = np.array([2, 4, 8])
    x_true = np.array([23/7, 8/7])
    x_test = solve_least_squares(A, b)
    assert np.allclose(x_true, x_test)

def test_eigenvalues(A):
    eig_val, eig_vec = get_eigenvalues(A)
    n = A.shape[0]
    for i in range(n):
        v0 = eig_val[i] * eig_vec[:,i]
        v1 = np.dot(A, eig_vec[:,i])
        assert np.allclose(v0, v1)

def test_all_eigen():
    test_eigenvalues(np.array([[0, -1], [-1, -3]], dtype = np.float64))
    # test_eigenvalues(np.array([[1, 2, 0], [-2, 1, 3], [0, -3, 1]], dtype = np.float64))
    for n in range(2, 9):
        A = np.random.rand(n, n)
        test_eigenvalues(A + A.T)


test_QR_decomposition()
test_QR_solve()
test_least_squares()
test_all_eigen()
print("All tests passed!")