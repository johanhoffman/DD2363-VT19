import numpy as np

def normalize(v):
    return v / np.linalg.norm(v)

def norm(v):
    return np.linalg.norm(v)

def jacobi_iterate(A, b):
    D = np.diag(np.diag(A))
    R = A - D
    D_inv = np.diag(1/np.diag(D))
    x = np.ones(A.shape[1])
    for _ in range(1000):
        x = np.dot(D_inv, b - np.dot(R, x))
    return x

def gauss_seidel(A, b):
    x = np.zeros(A.shape[1])
    for _ in range(1000):
        x_new = x.copy()
        for i in range(x.size):
            L_sum = np.dot(A[i,:i], x_new[:i])
            U_sum = np.dot(A[i,i+1:], x[i+1:])
            x_new[i] = 1/A[i,i]*(b[i] - L_sum - U_sum)
            x = x_new
    return x_new

def df(f, x, h):
    return (f(x+h) - f(x-h)) / (2*h)

def newton(f, x):
    iters = 0
    fx = f(x)
    while np.max(abs(fx)) > 1e-15:
        fx = f(x)
        x -= fx / df(f, x, 1e-8)
        iters += 1
        if iters > 100000:
            print(f(x))
            raise ValueError("No solution found")
    return x

def arnoldi(A, b, k, Q, H, k_max):
    v = np.dot(A, Q[:,k])
    for j in range(k):
        H[j,k] = np.dot(Q[:,j], v)
        v -= H[j,k]*Q[:,j]
    H[k+1,k] = norm(v)
    if (H[k+1, k] != 0 and k != k_max -1):
        Q[:,k+1] = v / H[k+1,k]

def gmres(A, b):
    k_max = 100
    n = A.shape[0]
    x0 = np.zeros(n)

    x = np.zeros(n)
    r = b - np.dot(A, x0)

    Q = np.zeros((n, k_max))
    H = np.zeros((k_max+1, k_max))
    Q[:,0] = normalize(r)

    for k in range(k_max):
        arnoldi(A, b, k, Q, H, k_max)

        b = np.zeros(k_max+1)
        b[0] = norm(r)

        res = np.linalg.lstsq(H, b, rcond = None)[0]
        x = np.dot(Q, res) + x0

    return x


def test_jacobi_iteration():
    A = np.array([[2, 1], [5, 7]], dtype = np.float64)
    b = np.array([11, 13])
    x_true = np.array([7+1/9, -3-2/9])
    x = jacobi_iterate(A, b)
    assert np.allclose(x, x_true)
    A = np.array(
        [
            [10, -1, 2, 0], 
            [-1, 11, -1, 3], 
            [2, -1, 10, -1], 
            [0, 3, -1, 8]
        ], dtype = np.float64
    )
    b = np.array([6, 25, -11, 15])
    x_true = np.array([1, 2, -1, 1])
    x = jacobi_iterate(A, b)
    assert np.allclose(x, x_true)

def test_gauss_seidel():
    A = np.array([[16, 3], [7, -11]], dtype = np.float64)
    b = np.array([11, 13])
    x_true = np.array([0.81218274, -0.66497462])
    x = gauss_seidel(A, b)
    assert np.allclose(x, x_true)
    A = np.array([[2, 1], [5, 7]], dtype = np.float64)
    b = np.array([11, 13])
    x_true = np.array([7+1/9, -3-2/9])
    x = gauss_seidel(A, b)
    assert np.allclose(x, x_true)

def test_polynomial():
    for _ in range(100):
        p = np.random.randn()
        q = np.random.randn()
        f = lambda x: x**2 + p*x + q
        if q < p*p/4:
            x_0 = -p/2 + (p*p/4 - q)**.5
            x_1 = -p/2 - (p*p/4 - q)**.5
            x_start = 0
            x = newton(f, x_start)
            assert np.isclose(x, x_0) or np.isclose(x, x_1)

def test_GMRES():
    A = np.array([[2, 1], [5, 7]], dtype = np.float64)
    b = np.array([11, 13])
    x_true = np.array([7+1/9, -3-2/9])
    # x = GMRes(A, b, np.zeros(A.shape[1]), 0, 5)
    # print(x)
    x = gmres(A, b)
    print(x)
    print(x_true)
    assert np.allclose(x, x_true)
    

def test_vector_polynomial():
    f = lambda x: (x - 1)**2 
    g = lambda x: (x- np.arange(x.size))**3
    for n in range(1, 10):
        x = np.zeros(n)
        x = newton(f, x)
        assert np.allclose(x, 1)
        x = np.zeros(n)
        x = newton(g, x)        
        assert np.allclose(x, np.arange(x.size))



test_jacobi_iteration()
test_gauss_seidel()
test_polynomial()
test_GMRES()
test_vector_polynomial()
print("All tests passed!")