import unittest
from lab3 import *

class Lab3FunctionsTest(unittest.TestCase):

    def test_jacobi_iteration(self):
        TOL_list = [1, 1e-2, 1e-4, 1e-7]
        A = np.array([ # diagonally dominat matrix
                    [10.,1.,-2.,3.],
                    [-1.,15.,4.,1.],
                    [0.,0.,4.,-1.],
                    [10.,-7.,2.,80.]
                    ])
        b = np.array([0.,4.,-2.,1.])
        for TOL in TOL_list:
            x = jacobi_iteration(A,b,TOL)
            self.assertTrue(np.linalg.norm(A.dot(x)-b) < TOL)
        A2 = np.array([[3,0,-1], [0,-2,1], [0,0,6]])
        b2 = np.array([4,0,1])
        y = np.array([25./18., 1./12., 1./6.]) # exact solution
        x2 = jacobi_iteration(A2,b2,TOL_list[3])
        self.assertTrue(np.linalg.norm(x2-y) < TOL_list[3])

    def test_gauss_seidel_iteration(self):
        TOL_list = [1, 1e-2, 1e-4, 1e-7]
        A = np.array([ # diagonally dominant matrix
                    [10.,1.,-2.,3.],
                    [-1.,15.,4.,1.],
                    [0.,0.,4.,-1.],
                    [10.,-7.,2.,80.]
                    ])
        b = np.array([0.,4.,-2.,1.])
        for TOL in TOL_list:
            x = gauss_seidel_iteration(A,b,TOL)
            self.assertTrue(np.linalg.norm(A.dot(x)-b) < TOL)
        A2 = np.array([[3,0,-1], [0,-2,1], [0,0,6]])
        b2 = np.array([4,0,1])
        y = np.array([25./18., 1./12., 1./6.]) # exact solution
        x2 = gauss_seidel_iteration(A2,b2,TOL_list[3])
        self.assertTrue(np.linalg.norm(x2-y) < TOL_list[3])

    def test_newtons_method(self):
        f = lambda x: x**3 - 2*x - 4
        df = lambda x: 3*x**2 - 2
        TOL_list = [1, 1e-2, 1e-4, 1e-7]
        for TOL in TOL_list:
            x = newtons_method(f,df,TOL)
            self.assertTrue(f(x) < TOL)
            self.assertTrue(np.linalg.norm(f(x)-f(2.)) < TOL)

    def test_gmres(self):
        A = np.array([[1,0,-2,4],[-1,2,5,3], [-7,8,2,9], [0,8,2,-1]])
        b = np.array([1,-2,5,3])
        TOL_list = [1, 1e-2, 1e-4, 1e-7]
        for TOL in TOL_list:
            x = gmres(A,b,TOL)
            self.assertTrue(np.linalg.norm(b-A.dot(x)) < TOL)
        A2 = np.array([[-2,3],[8,2]])
        b2 = np.array([1,-1])
        y = np.array([-10./56., 3./14.])
        for TOL in TOL_list:
            x2 = gmres(A2,b2,TOL)
            self.assertTrue(np.linalg.norm(y-x2) < TOL)

    def test_newtons_systems(self):
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
        TOL_list = [1, 1e-2, 1e-4, 1e-7]
        for TOL in TOL_list:
            x = newtons_systems(f,df,3,TOL)
            self.assertTrue(np.linalg.norm(f(x)) < TOL)
        f2 = lambda x: np.array([x[0]-3,x[1]+2,x[2]])
        df2 = lambda x: np.array([[1,0,0], [0,1,0], [0,0,1]])
        y = np.array([3,-2,0]) # exact solution
        for TOL in TOL_list:
            x2 = newtons_systems(f2,df2,3,TOL)
            self.assertTrue(np.linalg.norm(x2-y) < TOL)
if __name__ == '__main__':
    unittest.main()
