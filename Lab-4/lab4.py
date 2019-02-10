import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri
from matplotlib import axes
from math import *

# arguments: x, point to calculate
#            x_pts: array of x coordinates
#            y_pts: array of y coordinates
# return:  lagrange polynomial approximation of x
def lagrange_interpolation(x,x_pts,y_pts):
    points = x_pts.size
    y_inter = 0.0
    for i in range(points):
        lambda_i = 1
        for j in range(points):
            if j != i:
                lambda_i *= (x-x_pts[j])/(x_pts[i]-x_pts[j])
        y_inter += y_pts[i]*lambda_i
    return y_inter

def one_point_gassuian_quad(f, a=0, b=1):
    return (b-a)*f((a+b)/2.)

#arguments: function f of single variable
# return:   one point gaussian quadrature for interval 0 <= x <= 1
# def one_point_gaussian_quad(f):
#     x_pts = np.array([0.0, 1.0])
#     y_pts = np.array([f(x) for x in x_pts], dtype=np.float32)
#     return lagrange_interpolation(0.5,x_pts,y_pts) # x = 0.5 is midpoint for interval

def matrix_assembly(x_pts):
    n = x_pts.size-1
    M = np.zeros((n+1,n+1))
    # rather than iterating over each element of M
    # iterate over each interval of the mesh
    for i in range(n):
        h_i = x_pts[i+1] - x_pts[i]
        M[i,i] += h_i/3.
        M[i,i+1] = h_i/6.
        M[i+1,i] = h_i/6.
        M[i+1,i+1] += h_i/3.
    return M

def load_vector_assembly(f,x_pts):
    n = x_pts.size-1
    b = np.zeros((n+1,1))
    for i in range(n):
        h_i = x_pts[i+1]-x_pts[i]
        b[i] += f(x_pts[i])*h_i/2.
        b[i+1] += f(x_pts[i+1])*h_i/2.
    return b

def L2_projection_pw_linear_approx_1D(f,x_pts):
    M = matrix_assembly(x_pts)
    b = load_vector_assembly(f,x_pts)
    return np.linalg.solve(M,b)

if __name__=='__main__':

    f = lambda x: np.sin(2*pi*x)
    x_pts = np.arange(0.0,1.2,0.2)
    approx_pts = L2_projection_pw_linear_approx_1D(f,x_pts)

    more_xpts = np.arange(0.0,1.0,0.01)
    exact_pts = np.array([f(x) for x in more_xpts], dtype=np.float32)

    plt.plot(x_pts,approx_pts,label='L2 Projection')
    plt.plot(more_xpts,exact_pts, label='Exact function')
    #plt.plot(x_pts,lagrange_pts, label='Lagrange interpolation')
    plt.legend()
    plt.show()
