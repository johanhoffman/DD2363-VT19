import unittest
from lab4 import *

class Lab4FunctionsTest(unittest.TestCase):

    def test_one_point_gaussian_quad(self):
        f1 = lambda x: x**2
        res = one_point_gassuian_quad(f1)
        #on interval [0,1] exact integral is 1/4 for x^2
        self.assertEquals(res,0.25)
        f2 = lambda x : 3*x-1
        res = one_point_gassuian_quad(f2)
        #for a linear function f the integral is exact
        self.assertEquals(res,0.5)

    def test_L2_projection_pw_linear_1D(self):
        f1 = lambda x: sin(x)*3
        steps = np.array([1.,0.5,0.25,0.125,0.0625])
        errors = np.zeros(steps.size)
        print errors
        for step in range(steps.size):
            x_pts = np.arange(0,11,steps[step])
            l2_res = L2_projection_pw_linear_approx_1D(f1,x_pts)
            lst_sq_res = sum([(f1(x_pts[i])-l2_res[i])**2 for i in range(x_pts.size)])
            errors[step] = lst_sq_res
            plt.plot(step,lst_sq_res, 'o',label="Step size = "+`steps[step]`)
        plt.plot(np.arange(0,steps.size),errors, label='Error convergenence')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    unittest.main()
