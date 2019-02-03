import numpy as np

def jacobi_iteration(A,b,TOL):
    n  = A.shape[0]
    x = np.zeros(n)
    r_norm = np.linalg.norm(b-A.dot(x))
    while r_norm >= TOL:
        x_old = np.copy(x)
        for i in range(n):
            val = 0.0
            for j in range(n):
                if j != i:
                    val = val + A[i,j]*x_old[j]
            x[i] = (b[i]-val)/A[i,i]
        r_norm = np.linalg.norm(b-A.dot(x))
    return x

def gauss_seidel_iteration(A,b,TOL):
    x = np.zeros(A.shape[0])
    n,m = A.shape
    r_norm = np.linalg.norm(b-A.dot(x))
    while r_norm >= TOL:
        x_next = np.zeros_like(x)
        for i in range(n):
            val = b[i]
            for j in range(n):
                if j < i:
                    val -= A[i,j]*x_next[j]
                if j > i:
                    val -= A[i,j]*x[j]
            x_next[i] = np.float(val/A[i,i])
        x = x_next.copy()
        r_norm = np.linalg.norm(b-A.dot(x_next))
    return x

def newtons_method(f,df,TOL,x=None):
    if x == None:
        x = np.random.rand()
    iteration = 0
    while np.linalg.norm(f(x)) >= TOL and iteration < 500:
        x -= f(x)/df(x)
    if iteration == 500:
        return None
    return x

def arnoldi_iteration(A,b,k):
    Q = np.zeros((A.shape[0],k+1))
    H = np.zeros((k+1,k))
    Q[:,0] = b/np.linalg.norm(b)
    for i in range(k):
        v = A.dot(Q[:,i])
        for j in range(i+1):
            H[j,i] = np.inner(Q[:,j],v)
            v -= H[j,i]*Q[:,j]
        H[i+1,i] = np.linalg.norm(v)
        Q[:,i+1] = v / H[i+1,i]
    return Q,H

def gmres(A,b,TOL):
    x = np.zeros(A.shape[0])
    b_norm = np.linalg.norm(b)
    k = 0
    while np.linalg.norm(b-A.dot(x)) >= TOL:
        Q,H = arnoldi_iteration(A,b,k) # this gives us Q_{k+1} and correspinding upper Hessenberg
        e1 = np.zeros(k+1)
        e1[0] = 1
        y = np.linalg.lstsq(H,b_norm*e1,rcond=-1)[0]
        x = Q[:,:k].dot(y)
        k = k + 1
    return x

def newtons_systems(f, jacobian, dim, TOL):
    x = np.random.rand(dim)
    while np.linalg.norm(f(x)) >= TOL:
        grad = gmres(jacobian(x),-f(x),TOL)
        x += grad
    return x

f = lambda x: np.array([
                        (x[0]**2)*x[1] + 3*x[2] - x[1]**3 + 1,
                        3*x[1]-x[2]**2+x[0],
                        x[0]**3-7*x[1]+x[2]
                      ])
df = lambda x: np.array([
                    [ 2*x[0]*x[1], x[0]**2-3*x[1]**2, 3 ],
                    [ 1, 3, -2*x[2] ],
                    [3*x[0]**2, -7, 1 ]
                    ])

if __name__ == '__main__':
    A = np.array([[2,0,5], [4,-1,3], [-2,9,1]])
    b = np.array([1,-2,8])
    x = newtons_systems(f,df,3,1e-10)
    print x
