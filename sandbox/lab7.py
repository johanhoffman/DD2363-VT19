import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gradient_descent(f, df, x):
	a = .01
	conv = []
	x_trail = [x.copy()]
	while np.linalg.norm(df(x)) > 1e-6:
		x -= a*df(x)
		conv.append(np.linalg.norm(x-1))
		x_trail.append(x.copy())

	# conv = np.array(conv)
	# conv = np.linalg.norm(conv - 1)
	return x, np.array(x_trail), conv

def test_gradient_descent_2D():
	X = np.arange(-5, 5, 0.25)
	Y = np.arange(-5, 5, 0.25)
	X, Y = np.meshgrid(X, Y)
	f_plot = lambda x, y: (x-1)**4 + (y-1)**4
	f = lambda x: np.dot((x-1)**2, (x-1)**2)
	df = lambda x: 4*(x-1)**3

	x_0 = np.array([-4.5, -4.0])
	x_min, x_trail, conv = gradient_descent(f, df, x_0)

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	Z = f_plot(X, Y)
	surf = ax.plot_wireframe(X, Y, Z,
		linewidth=.5, antialiased=True)
	ax.plot(x_trail[:,0], x_trail[:,1], f_plot(x_trail[:,0], x_trail[:,1]), 'ro')
	ax.set_xlim([-5, 5])
	ax.set_ylim([-5, 5])
	# fig.colorbar(surf, shrink=0.5, aspect=5)
	print(len(conv))
	plt.show()

def test_gradient_descent():
	f = lambda x: np.dot((x-1)**2, (x-1)**2)
	df = lambda x: 4*(x-1)**3
	x_0 = np.zeros(2)
	x_min, conv = gradient_descent(f, df, x_0)
	print(x_min)
	# print(conv)
	# plt.semilogy(conv, label = "Gradient descent")
	plt.loglog(conv, label = "Gradient descent")
	# plt.show()


def newtons_method(f, H, df, x):
	conv = []
	a = 0.01
	x_trail = []
	while np.linalg.norm(df(x)) > 1e-12:
		x_trail.append(x.copy())
		x -= a*np.linalg.lstsq(H(x), df(x), rcond = None)[0]
		conv.append(np.linalg.norm(x-1))
	return x, np.array(x_trail), conv

def test_newtons_method_2D():
	X = np.arange(-5, 5, 0.25)
	Y = np.arange(-5, 5, 0.25)
	X, Y = np.meshgrid(X, Y)
	f_plot = lambda x, y: (x-1)**4 + (y-1)**4
	f = lambda x: np.dot((x-1)**2, (x-1)**2)
	df = lambda x: 4*(x-1)**3
	H = lambda x: np.diag(12*(x-1)**2)

	x_0 = np.array([-4.5, -4.0])
	x_min, x_trail, conv = newtons_method(f, H, df, x_0)

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	Z = f_plot(X, Y)
	surf = ax.plot_wireframe(X, Y, Z,
		linewidth=.5, antialiased=True)
	ax.plot(x_trail[:,0], x_trail[:,1], f_plot(x_trail[:,0], x_trail[:,1]), 'ro')
	print(len(conv))
	plt.show()

def test_newtons_method():
	# f = lambda x: np.dot(x-1, x-1)
	# df = lambda x: 2*(x-1)
	# H = lambda x: np.diag(0*x+2)
	f = lambda x: np.dot((x-1)**2, (x-1)**2)
	df = lambda x: 4*(x-1)**3
	H = lambda x: np.diag(12*(x-1)**2)
	x_0 = np.zeros(2)

	x_min, x_trail, conv = newtons_method(f, H, df, x_0)
	print(x_min)
	# print(conv)

	# plt.semilogy(conv, label = "Newton's method")
	plt.loglog(conv, label = "Newton's method")
	plt.legend()
	plt.show()


# test_gradient_descent_2D()
# test_gradient_descent()
# test_newtons_method()
test_newtons_method_2D()
