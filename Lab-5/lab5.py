import numpy as np
from math import *
from matplotlib import pyplot as plt
from matplotlib import tri
from matplotlib import axes

def explicit_euler(T,f,u0,N):
    h = T/N
    u = np.zeros(N+1)
    u[0] = u0
    t = np.zeros(N+1)
    t[0] = 0
    for n in range(1,N+1):
        t[n] = t[n-1]+h
        u[n] = u[n-1] + h*f(u[n-1],t[n-1])
    return u,t,h

def explicit_euler_system(T,f,u0,N):
    h = T/N
    u = np.zeros((u0.shape[0],N+1))
    u[:,0] = u0
    t = np.zeros(N+1)
    for n in range(1,N+1):
        t[n] = t[n-1]+h
        u[:,n] = u[:,n-1] + h*f(u[:,n-1],t[n-1])
    return u,t,h
