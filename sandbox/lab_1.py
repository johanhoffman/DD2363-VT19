import numpy as np

def inner_product(x, y):
	if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
		raise TypeError("Both arguments must be lists")
	if x.size != y.size:
		raise ValueError("Vectors must be same length")
	return (x*y).sum()

def inner_product_list(x, y):
	if not (isinstance(x, list) and isinstance(y, list)):
		raise TypeError("Both arguments must be lists")
	if len(x) != len(y):
		raise ValueError("Vectors must be same length")
	return sum(a*b for a, b in zip(x, y))

def inner_product_matrix(A, B):
	if not (isinstance(A, np.ndarray) and isinstance(B, np.ndarray)):
		raise TypeError("Both arguments must be lists")

	C = np.zeros((A.shape[0], B.shape[1]))
	for c in range(B.shape[1]):
		for r in range(A.shape[0]):
			C[r, c] = (A[r,:]*B[:,c]).sum()
	return C

A = np.random.rand(5, 3)
B = np.random.rand(3, 4)
print(np.dot(A, B))
print(inner_product_matrix(A, B))
x = np.random.rand(5)
y = np.random.rand(5)
print(x)
print(y)
print(np.dot(x, y.T))
print(inner_product(x, y))
print(inner_product_list(list(x), list(y)))
