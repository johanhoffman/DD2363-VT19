import numpy as np
from math import *

def barzilai_borwein(x_prev,x_curr,grad_f,alpha,beta):
    delta_grad = grad_f(x_curr) - grad_f(x_prev)
    delta_x = x_curr - x_prev
    denom = delta_grad.dot(delta_grad)
    nom = delta_grad.dot(delta_x)
    if denom == 0:
        return alpha*beta
    return nom/denom

def gradient_descent(f,grad_f,x_current,TOL):
    alpha = 1 # initial stepsize
    beta = 0.9
    max_iter = 1000
    iter = 0
    while np.linalg.norm(grad_f(x_current)) >= TOL and iter < max_iter:
        old_x = x_current
        x_current = old_x - alpha*grad_f(old_x)
        alpha = barzilai_borwein(old_x,x_current,grad_f,alpha,beta)
        iter += 1
    return x_current

def newtons_method(f,grad_f,Hf,x_curr,TOL):
    while np.linalg.norm(grad_f(x_curr)) >= TOL:
        x_delta = np.linalg.solve(Hf(x_curr),-grad_f(x_curr))
        x_curr += x_delta
    return x_curr

if __name__ == "__main__":
    f = lambda x: x[0]**2 + x[1]**2 -3
    grad_f = lambda x: np.array([2.*x[0],2.*x[1]])
    Hf = lambda x: np.array([
                    [2.0, 0.0],
                    [0.0, 2.0]
                ])
    print f(newtons_method(f,grad_f,Hf,2,1e-6))
