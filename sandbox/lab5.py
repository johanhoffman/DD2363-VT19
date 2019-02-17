import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

def euler_forward(f, u_0, T, dt):
    t = [0]
    u = [u_0]
    while t[-1] < T:
        du = f(u[-1], t[-1])
        u.append(u[-1] + du*dt)
        t.append(t[-1] + dt)
    return np.array(t), np.array(u)

def rk_2(f, u_0, T, dt):
    t = [0]
    u = [u_0]
    du = f(u[-1], t[-1])
    while t[-1] < T:
        new_du = f(u[-1] + du*dt, t[-1] + dt)
        u.append(u[-1] + .5*(du + new_du)*dt)
        t.append(t[-1] + dt)
        du = new_du
    return np.array(t), np.array(u)

def rk_4(f, u_0, T, dt):
    t = [0]
    u = [u_0]
    while t[-1] < T:
        k1 = dt * f(u[-1], t[-1])
        k2 = dt * f(u[-1] + k1/2, t[-1] + dt/2)
        k3 = dt * f(u[-1] + k2/2, t[-1] + dt/2)
        k4 = dt * f(u[-1] + k3, t[-1] + dt)
        u.append(u[-1] + 1/6*(k1 + 2*k2 + 2*k3 + k4))
        t.append(t[-1] + dt)
    return np.array(t), np.array(u)

def test_error_over_time_1D():
    f = lambda u, t: 10 - u*u
    sol = lambda t: ((10**.5) * (np.exp(2 * (10**.5) * t) - 1))/(np.exp(2 * (10**.5) * t) + 1)
    u_0 = 0
    fig, ax = plt.subplots(1, 3, sharey = True)
    for i in range(3, 12):
        t, u = euler_forward(f, u_0, 4, 2**-i)
        err = abs(u - sol(t))
        ax[0].semilogy(t, err, '-')
        t, u = rk_2(f, u_0, 4, 2**-i)
        err = abs(u - sol(t))
        ax[1].semilogy(t, err, '-')
        t, u = rk_4(f, u_0, 4, 2**-i)
        err = abs(u - sol(t))
        ax[2].semilogy(t, err, '-')
    plt.suptitle("Changes in error over time when doubling number of iteration points")
    ax[0].set_title("Euler forward")
    ax[0].set_xlabel("Time(s)")
    ax[1].set_title("Runge-Kutta 2")
    ax[1].set_xlabel("Time(s)")
    ax[2].set_title("Runge-Kutta 4")
    ax[2].set_xlabel("Time(s)")
    ax[0].set_ylabel("Error")
    plt.show()

def test_error_over_time_2D():
    f = lambda u, t: np.dot(np.array([[113/19, -45/19], [30/19, 20/19]]), u)
    sol = lambda t: np.array([(3/19 * 3 * np.exp(2*t) + 2/19 * 5 * np.exp(5*t)), (3/19 * 5 * np.exp(2*t) + 2/19 *   2 * np.exp(5*t))])
    u_0 = np.array([1, 1])
    fig, ax = plt.subplots(1, 3, sharey = True)
    for i in range(11, 12):
        t, u = euler_forward(f, u_0, 4, 2**-i)
        err = np.linalg.norm(u - sol(t).T, axis = 1)
        ax[0].semilogy(t, err, '-')
        # ax[0].semilogy(t, sol(t).T, '-')
        t, u = rk_2(f, u_0, 4, 2**-i)
        err = np.linalg.norm(u - sol(t).T, axis = 1)
        ax[1].semilogy(t, err, '-')
        t, u = rk_4(f, u_0, 4, 2**-i)
        err = np.linalg.norm(u - sol(t).T, axis = 1)
        ax[2].semilogy(t, err, '-')
    plt.suptitle("Changes in error over time when doubling number of iteration points")
    ax[0].set_title("Euler forward")
    ax[0].set_xlabel("Time(s)")
    ax[1].set_title("Runge-Kutta 2")
    ax[1].set_xlabel("Time(s)")
    ax[2].set_title("Runge-Kutta 4")
    ax[2].set_xlabel("Time(s)")
    ax[0].set_ylabel("Error")
    plt.show()


def test_convergence_rate_1D():
    plt.figure()
    f = lambda u, t: 10 - u*u
    sol = lambda t: ((10**.5) * (np.exp(2 * (10**.5) * t) - 1))/(np.exp(2 * (10**.5) * t) + 1)
    u_0 = 0

    err_euler = []
    err_rk2 = []
    err_rk4 = []
    iters = []
    for i in range(3, 15):
        t, u = euler_forward(f, u_0, 4, 2**-i)
        err_euler.append(np.abs(u - sol(t)).max())
        t, u = rk_2(f, u_0, 4, 2**-i)
        err_rk2.append(np.abs(u - sol(t)).max())
        t, u = rk_4(f, u_0, 4, 2**-i)
        err_rk4.append(np.abs(u - sol(t)).max())
        iters.append(len(t))

    plt.loglog(iters, err_euler, 'r', label = "Euler forward")
    plt.loglog(iters, err_rk2, 'g', label = "Runge-Kutta 2")
    plt.loglog(iters, err_rk4, 'b', label = "Runge-Kutta 4")
    c_euler = (np.log(err_euler[5]) - np.log(err_euler[0]))/(np.log(iters[5]) - np.log(iters[0]))
    c_rk2 = (np.log(err_rk2[5]) - np.log(err_rk2[0]))/(np.log(iters[5]) - np.log(iters[0]))
    c_rk4 = (np.log(err_rk4[5]) - np.log(err_rk4[0]))/(np.log(iters[5]) - np.log(iters[0]))
    plt.text(1*10**3, 1.5*10**-2, "$q = %.2f$" % c_euler, rotation = -7)
    plt.text(1*10**3, 2*10**-4, "$q = %.2f$" % c_rk2, rotation = -17)
    plt.text(1*10**3, 2*10**-9, "$q = %.2f$" % c_rk4, rotation = -35)
    plt.xlabel("Iterations")
    plt.ylabel("Maximum error")
    plt.title("Approximation error for system $\\frac{du}{dt} = 10 - u^2$")
    plt.legend()
    plt.show()

def test_convergence_rate_2D():
    plt.figure()
    f = lambda u, t: np.dot(np.array([[-3, -2], [-2, -3]]), u)
    sol = lambda t: np.array([(2 * np.exp(-5*t) - 1 * np.exp(-t)), (2 * np.exp(-5*t) + 1 * np.exp(-t))])
    u_0 = np.array([1, 3])

    err_euler = []
    err_rk2 = []
    err_rk4 = []
    iters = []
    for i in range(3, 12):
        t, u = euler_forward(f, u_0, 4, 2**-i)
        err_euler.append(np.linalg.norm(u - sol(t).T, axis = 1).max())
        t, u = rk_2(f, u_0, 4, 2**-i)
        err_rk2.append(np.linalg.norm(u - sol(t).T, axis = 1).max())
        t, u = rk_4(f, u_0, 4, 2**-i)
        err_rk4.append(np.linalg.norm(u - sol(t).T, axis = 1).max())
        iters.append(len(t))

    plt.loglog(iters, err_euler, 'r', label = "Euler forward")
    plt.loglog(iters, err_rk2, 'g', label = "Runge-Kutta 2")
    plt.loglog(iters, err_rk4, 'b', label = "Runge-Kutta 4")
    c_euler = (np.log(err_euler[5]) - np.log(err_euler[0]))/(np.log(iters[5]) - np.log(iters[0]))
    c_rk2 = (np.log(err_rk2[5]) - np.log(err_rk2[0]))/(np.log(iters[5]) - np.log(iters[0]))
    c_rk4 = (np.log(err_rk4[5]) - np.log(err_rk4[0]))/(np.log(iters[5]) - np.log(iters[0]))
    plt.text(1*10**3, 2*10**-2, "$q = %.2f$" % c_euler, rotation = -7)
    plt.text(1*10**3, 3*10**-4, "$q = %.2f$" % c_rk2, rotation = -14)
    plt.text(1*10**3, 3*10**-9, "$q = %.2f$" % c_rk4, rotation = -28)
    plt.xlabel("Iterations")
    plt.ylabel("Maximum error")
    plt.title(r"Approximation error for system $\frac{d\vec{u}}{dt} = A \cdot \vec{u}$")
    plt.legend()
    plt.show()


def test_compare_errors_1D():
    plt.figure()
    f = lambda u, t: 10 - u*u
    sol = lambda t: ((10**.5) * (np.exp(2 * (10**.5) * t) - 1))/(np.exp(2 * (10**.5) * t) + 1)
    u_0 = 0

    t, u = euler_forward(f, u_0, 4, 2**-12)
    err = abs(u - sol(t))
    plt.semilogy(t, err, 'b', label = "Euler forward")
    t, u = rk_2(f, u_0, 4, 2**-12)
    err = abs(u - sol(t))
    plt.semilogy(t, err, 'r', label = "Runge-Kutta 2")
    t, u = rk_4(f, u_0, 4, 2**-12)
    err = abs(u - sol(t))
    plt.semilogy(t, err, 'g', label = "Runge-Kutta 4")
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.title("Approximation error for system $\\frac{du}{dt} = 10 - u^2$")
    plt.legend()
    plt.show()

def test_compare_errors_2D():
    plt.figure()
    f = lambda u, t: np.dot(np.array([[-3, -2], [-2, -3]]), u)
    sol = lambda t: np.array([(2 * np.exp(-5*t) - 1 * np.exp(-t)), (2 * np.exp(-5*t) + 1 * np.exp(-t))])
    u_0 = np.array([1, 3])

    t, u = euler_forward(f, u_0, 4, 2**-12)
    err = np.linalg.norm(u - sol(t).T, axis = 1)
    plt.semilogy(t, err, 'b', label = "Euler forward")
    t, u = rk_2(f, u_0, 4, 2**-12)
    err = np.linalg.norm(u - sol(t).T, axis = 1)
    plt.semilogy(t, err, 'r', label = "Runge-Kutta 2")
    t, u = rk_4(f, u_0, 4, 2**-12)
    err = np.linalg.norm(u - sol(t).T, axis = 1)
    plt.semilogy(t, err, 'g', label = "Runge-Kutta 4")
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.title(r"Approximation error for system $\frac{d\vec{u}}{dt} = A \cdot \vec{u}$")
    plt.legend()
    plt.show()

def f_n_body(u, t):
    M = np.array([
        1.989e30, 
        5.972e24, 
        7.34767309e22, 
        3.825e23,
        4.867e24
    ])
    du = np.zeros_like(u)
    n = u.shape[0]//2 
    G = 6.67e-11
    du[n:,:] = u[:n,:]
    for i in range(n):
        for j in range(i+1, n):
            diff = u[n+j,:] - u[n+i,:]
            r = np.linalg.norm(diff)
            direction = diff / r
            # apply acceleration to pair of bodies
            du[i,:] += direction * G * M[j] / r**2
            du[j,:] -= direction * G * M[i] / r**2

    return du

def n_body_problem():
    v0 = np.array([
        [0, 0, 0], # sun
        [1.08e5/3.6*np.cos(7.155*np.pi/180), 0, 1.08e5/3.6*np.sin(7.155*np.pi/180)], # earth
        [1.08e5/3.6*np.cos(7.155*np.pi/180) + 3.683e3/3.6, 0, 1.08e5/3.6*np.sin(7.155*np.pi/180)], # moon
        [1.70505e5/3.6*np.cos(3.38*np.pi/180), 0, 1.70505e5/3.6*np.sin(3.38*np.pi/180)], #mercury
        [1.26077e5/3.6*np.cos(3.86*np.pi/180), 0, 1.26077e5/3.6*np.sin(3.86*np.pi/180)] # venus
    ])
    x0 = np.array([
        [0, 0, 0],
        [0, -1.496e11, 0],
        [0, -1.496e11 - 3.844e8, 0],
        [0, -5.791e10, 0],
        [0, -1.082e11, 0]
    ])
    n = x0.shape[0]
    u0 = np.zeros((2*n, 3))
    u0[:n,:] = v0
    u0[n:,:] = x0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    t_e, u_e = euler_forward(f_n_body, u0.copy(), 86400*365*1, 8640)
    t_2, u_2 = rk_2(f_n_body, u0.copy(), 86400*365*1, 8640)
    t_4, u_4 = rk_4(f_n_body, u0.copy(), 86400*365*1, 8640)
    for i in range(n):
        x_e, y_e, z_e = u_e[:,n+i,:].T
        x_2, y_2, z_2 = u_2[:,n+i,:].T
        x_4, y_4, z_4 = u_4[:,n+i,:].T
        ax.plot(x_e, y_e, z_e - 3e11, c = (0, 0, (i+1)/n))
        ax.plot(x_2, y_2, z_2, c = ((i+1)/n, 0, 0))
        ax.plot(x_4, y_4, z_4 + 3e11, c = (0, (i+1)/n, 0))
    ax.set_xlim([-3e11, 3e11])
    ax.set_ylim([-3e11, 3e11])
    ax.set_zlim([-3e11, 3e11])
    ax.set_aspect('equal')
    plt.title("Simulation of n-body problem for our solar system")
    plt.show()


def test_ode_solvers():
    # plt.ion()
    # test_error_over_time_1D()
    # test_convergence_rate_1D()
    # test_compare_errors_1D()

    test_error_over_time_2D()
    # test_convergence_rate_2D()
    # plt.ioff()
    # test_compare_errors_2D()


rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

test_ode_solvers()
# n_body_problem()

