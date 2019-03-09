import unittest
from lab7 import *

class Lab7FunctionsTest(unittest.TestCase):

    def test_gradient_descent(self):
        f = lambda x: (x[0]-1.0)**2 + (x[1]+2.0)**2 - 3.0
        # minimum of f is at (1,-2) and f(1,-2) = -3
        grad_f = lambda x: np.array([2.0*(x[0]-1.0), 2.0*(x[1]+2.0)])
        TOL_list = [.1,.01,.001,.0001,.00001,.000001,.0000001]
        rel_error = []
        exact = np.array([1.0,-2.0])
        for TOL in TOL_list:
            x_curr = gradient_descent(f,grad_f,np.array([10.0,3.0]),TOL)
            rel_error.append(np.linalg.norm(x_curr-exact)/np.linalg.norm(exact))
        print("Results for Gradient Descent")
        print ("The relative error for step sizes:")
        for i in range(len(TOL_list)):
            print("Step size: %f | Error: %f" %(TOL_list[i],rel_error[i]))

    def test_newtons_method(self):
        f = lambda x: (x[0]-2.0)**2 + (x[1]+1.0)**2 -3
        grad_f = lambda x: np.array([2.*(x[0]-2.0),2.0*(x[1]+1.0)])
        Hf = lambda x: np.array([
                    [2.0, 0.0],
                    [0.0, 2.0]
                ])
        TOL_list = [.1,.01,.001,.0001,.00001,.000001,.0000001]
        rel_error = []
        exact = np.array([2.0,-1.0])
        for TOL in TOL_list:
            x_curr = newtons_method(f,grad_f,Hf,np.array([10.0,3.0]),TOL)
            rel_error.append(np.linalg.norm(x_curr-exact)/np.linalg.norm(exact))
        print("Results for Newton's method")
        print ("The relative error for step sizes:")
        for i in range(len(TOL_list)):
            print("Step size: %f | Error: %f" %(TOL_list[i],rel_error[i]))
if __name__ == '__main__':
    unittest.main()
