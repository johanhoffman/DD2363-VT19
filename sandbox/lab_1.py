import numpy as np

class CRF:

	def __init__(self, A):
		self.make_CRF(A)

	def make_CRF(self, A):
		self.calc_val(A)
		self.calc_col_idx(A)
		self.calc_row_ptr(A)
		self.make_one_indexed()

	def calc_val(self, A):
		self.val = A[A > 0].flatten()

	def calc_col_idx(self, A):
		col_idx = np.tile(np.arange(A.shape[1]), (A.shape[0], 1))
		self.col_idx = col_idx[A > 0].flatten()		

	def calc_row_ptr(self, A):
		row_idx = np.tile(np.arange(A.shape[0]), (A.shape[1], 1)).T
		row_indices = row_idx[A > 0].flatten()
		diffs = np.diff(row_indices)
		diff_idx = diffs * (np.arange(diffs.size) + 1)
		filtered_indices = diff_idx[diff_idx > 0]
		print("row_indices:", row_indices)
		print("diffs:", diffs)
		print("diff_idx:", diff_idx)
		self.row_ptr = np.zeros(filtered_indices.size + 1, dtype = np.int32)
		self.row_ptr[1:] = filtered_indices

	def make_one_indexed(self):
		self.col_idx += 1
		self.row_ptr += 1


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

sparse_matrix = np.array(
	[[3, 2, 0, 2, 0, 0],
	 [0, 2, 1, 0, 0, 0],
	 [0, 0, 1, 0, 0, 0],
	 [0, 0, 3, 2, 0, 0],
	 [0, 0, 0, 0, 1, 0],
	 [0, 0, 0, 0, 2, 3]]
)

A_CRF = CRF(sparse_matrix)
print(A_CRF.val, A_CRF.val == [3, 2, 2, 2, 1, 1, 3, 2, 1, 2, 3])
print(A_CRF.col_idx, A_CRF.col_idx == [1, 2, 4, 2, 3, 3, 3, 4, 5, 5, 6])
print(A_CRF.row_ptr, A_CRF.row_ptr == [1, 4, 6, 7, 9, 10])
quit()
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
