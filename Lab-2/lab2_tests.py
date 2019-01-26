import unittest
from lab2 import *

class Lab2FunctionsTest(unittest.TestCase):

    def test_qr_factorization(self):
        # randomly generate a square matrix
        A = np.random.rand(3,3)
        Q,R = gs(A)
        Qt = np.transpose(Q)
        # tranpose(Q) = inverse(Q) -> tranpose(Q) * Q = I
        # so Frobenius norm of : tranpose(Q) * Q - I = 0
        self.assertTrue(np.linalg.norm(Qt.dot(Q)-np.identity(3)) < 0.000001)
        # A = QR -> QR - A = 0-matrix -> Frobenius norm of QR-A = 0
        self.assertTrue(np.linalg.norm(Q.dot(R)-A) < 0.000001)

    def test_direct_solver(self):
        A = np.random.rand(3,3)
        b = np.random.rand(3)
        x = direct_solver(A,b)
        # if x is a solution to Ax = b, then Ax-b = 0
        self.assertTrue(np.linalg.norm(A.dot(x)-b) < 0.0000001)

        # pre calculated test
        A2 = np.array([[1,2],[4,-2]])
        y = np.array([1.4,0.8])
        b2 = np.array([3,4])
        x2 = direct_solver(A2,b2)
        # assert that the numerical solution x2 is not far from the "exact" solution y2 |x-u| < eps
        self.assertTrue(np.linalg.norm(x2-y) < 0.0000001)

    def test_least_squares(self):
        A = np.array([[1,2], [4,2], [-1,0]])
        b = np.array([4,-2,3])
        x = least_squares(A,b)
        # x_np is numpy's least squares solution (np.linalg.lstsq)
        x_np = np.linalg.lstsq(A,b,rcond=-1)[0]
        self.assertTrue(np.linalg.norm(x-x_np) < 0.00000001)

        # A test for a randomized overdetermined system
        A_rand = np.random.rand(10,4)
        b_rand = np.random.rand(10)
        x_rand = least_squares(A_rand, b_rand)
        x_np_rand = np.linalg.lstsq(A_rand,b_rand,rcond=-1)[0]
        self.assertTrue(np.linalg.norm(x_rand-x_np_rand) < 0.00000001)
if __name__ == '__main__':
    unittest.main()
