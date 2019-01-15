import numpy as np
import time

class CRS:

	def __init__(self, A):
		self.one_indexed = False
		self.index_dtype = np.dtype("int64")
		self.make_CRS(A)

	def make_CRS(self, A):
		self.shape = A.shape
		self.dtype = A.dtype
		self.calc_val(A)
		self.calc_col_idx(A)
		self.calc_row_ptr(A)
		self.make_one_indexed()

	@property
	def val(self):
		return self._val
	
	@property
	def col_idx(self):
		return self._col_idx + self.one_indexed

	@property
	def row_ptr(self):
		return self._row_ptr + self.one_indexed

	def calc_val(self, A):
		self._val = A[A > 0].flatten()

	def calc_col_idx(self, A):
		# generate matrix the same size as A, where a_ij = j
		col_idx = np.tile(np.arange(A.shape[1]), (A.shape[0], 1))
		self._col_idx = col_idx[A > 0].flatten()

	def calc_row_ptr(self, A):
		"""Calculates the values of the row_ptr array in the CRS"""
		# generate matrix the same size as A, where a_ij = i
		row_idx = np.tile(
			np.arange(A.shape[0], dtype = self.index_dtype), 
			(A.shape[1], 1)
		).T
		# extract the row indices where A is non-zero
		row_indices = row_idx[A > 0].flatten()
		# the differences of row_indices indicate where a new row begins
		diffs = np.diff(row_indices)
		# to correctly handle empty rows, we must use this
		reverse_bincount = np.repeat(np.arange(diffs.size), diffs)
		row_sums = A.sum(axis = 1)
		row_cumsum = row_sums.cumsum()
		empty_top_rows = (row_cumsum == 0).sum()

		# populate the row_ptr array
		self._row_ptr = np.zeros(self.shape[0], dtype = self.index_dtype)
		start_index = empty_top_rows + 1
		end_index = reverse_bincount.size + 1 + empty_top_rows
		self._row_ptr[start_index:end_index] = reverse_bincount + 1
		# make sure that empty rows at the end are correctly reconstructed
		self._row_ptr[end_index:] = -1
		# self._row_ptr = np.array([0, 0, 3, -1])

	def make_one_indexed(self):
		"""Transform col_idx and row_ptr to use 1-indexing in output, 
		but not in the internal state"""
		self.one_indexed = True

	def print_stats(self):
		size = (self._val.size * self.dtype.itemsize 
			+ self._col_idx.size * self.index_dtype.itemsize
			+ self._row_ptr.size * self.index_dtype.itemsize)
		original_size = self.dtype.itemsize * self.shape[0] * self.shape[1]
		print("Space needed: %d bytes" % size)
		print("Original matrix size: %d bytes" % original_size)
		print("Compression ratio: %.1f%%" % (100*(1 - size / original_size),))

	def reconstruct(self):
		A = np.zeros(self.shape, dtype = self.dtype)
		row_starts = np.zeros(self._val.size, dtype = self.index_dtype)
		bbins = np.bincount(self._row_ptr[self._row_ptr >= 0])
		row_starts[:bbins.size] += bbins
		row_idx = row_starts.cumsum() - 1
		A[row_idx, self._col_idx] = self.val
		return A

	def __str__(self):
		return str(self.reconstruct())

	def __repr__(self):
		return self.reconstruct()

	def __mul__(self, x):
		return CRS.multiply(self, x)

	@staticmethod
	def multiply_slow(A, B):
		if isinstance(A, CRS):
			A = A.reconstruct()
		if isinstance(B, CRS):
			B = B.reconstruct()
		return np.dot(A, B)

	@staticmethod
	def multiply(A, B):
		if isinstance(A, CRS):
			row_starts = np.zeros(A._val.size, dtype = np.int64)
			bbins = np.bincount(A._row_ptr[A._row_ptr >= 0])
			row_starts[:bbins.size] += bbins
			row_idx = row_starts.cumsum() - 1
			res = np.zeros(A.shape[0])
			for i in range(A._val.size):
				res[row_idx[i]] += B[A._col_idx[i]] * A._val[i]
			return res
		elif isinstance(B, CRS):
			row_starts = np.zeros(B._val.size, dtype = np.int64)
			bbins = np.bincount(B._row_ptr[B._row_ptr >= 0])
			row_starts[:bbins.size] += bbins
			row_idx = row_starts.cumsum() - 1
			res = np.zeros(B.shape[1])
			for i in range(B._val.size):
				res[B._col_idx[i]] += A[row_idx[i]] * B._val[i]
			return res

		return np.dot(A, B)


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

	if B.ndim == 1:
		C = np.zeros((A.shape[0],))
		for r in range(A.shape[0]):
			C[r] = (A[r,:]*B).sum()
	else:
		C = np.zeros((A.shape[0], B.shape[1]))
		for c in range(B.shape[1]):
			for r in range(A.shape[0]):
				C[r, c] = (A[r,:]*B[:,c]).sum()
	return C

def test_inner_product():
	x = np.random.rand(5)
	y = np.random.rand(5)
	true_value = np.dot(x, y)
	np_test_value = inner_product(x, y)
	list_test_value = inner_product_list(list(x), list(y))
	assert true_value == np_test_value
	assert true_value == list_test_value

def test_matrix_vector_product():
	A = np.random.rand(5, 3)
	x = np.random.rand(3)
	true_value = np.dot(A, x)
	b = inner_product_matrix(A, x)
	assert np.array_equal(true_value, b)

def test_matrix_matrix_product():
	A = np.random.rand(5, 3)
	B = np.random.rand(3, 4)
	true_value = np.dot(A, B)
	C = inner_product_matrix(A, B)
	assert np.array_equal(true_value, C)

def test_known_CRS():
	sparse_matrix = np.array([
	 [3, 2, 0, 2, 0, 0],
	 [0, 2, 1, 0, 0, 0],
	 [0, 0, 1, 0, 0, 0],
	 [0, 0, 3, 2, 0, 0],
	 [0, 0, 0, 0, 1, 0],
	 [0, 0, 0, 0, 2, 3]])

	A_CRS = CRS(sparse_matrix)
	# A_CRS.print_stats()
	assert np.array_equal(A_CRS.val, [3, 2, 2, 2, 1, 1, 3, 2, 1, 2, 3])
	assert np.array_equal(A_CRS.col_idx, [1, 2, 4, 2, 3, 3, 3, 4, 5, 5, 6])
	assert np.array_equal(A_CRS.row_ptr, [1, 4, 6, 7, 9, 10])
	assert np.array_equal(A_CRS.reconstruct(), sparse_matrix)


def test_large_CRS(m, n):

	sparse_matrix = np.zeros((m, n))
	# create a tridiagonal matrix with random integers
	np.fill_diagonal(sparse_matrix, np.random.randint(0, 3, m))
	np.fill_diagonal(sparse_matrix[:,1:], np.random.randint(0, 3, m-1))
	np.fill_diagonal(sparse_matrix[1:,:], np.random.randint(0, 3, m-1))

	A_CRS = CRS(sparse_matrix)

	# assert that it can be properly reconstructed from its representation
	assert np.array_equal(A_CRS.reconstruct(), sparse_matrix)


def test_CRS_matrix():
	test_known_CRS()
	for m in range(2, 50):
		for n in range(2, 50):
			test_large_CRS(m, n)

def test_CRS_product_known():
	sparse_matrix = np.array([
	 [3, 2, 0, 2, 0, 0],
	 [0, 2, 1, 0, 0, 0],
	 [0, 0, 1, 0, 0, 0],
	 [0, 0, 3, 2, 0, 0],
	 [0, 0, 0, 0, 1, 0],
	 [0, 0, 0, 0, 2, 3]])

	A_CRS = CRS(sparse_matrix)
	t0 = time.clock()
	true_right_val = np.dot(sparse_matrix, [1, 2, 3, 4, 5, 6])
	true_left_val = np.dot([1, 2, 3, 4, 5, 6], sparse_matrix)
	t1 = time.clock()
	right_val = A_CRS * np.array([1, 2, 3, 4, 5, 6])
	left_val = CRS.multiply(np.array([1, 2, 3, 4, 5, 6]), A_CRS)
	t2 = time.clock()

	assert np.array_equal(true_right_val, right_val)
	assert np.array_equal(true_left_val, left_val)
	# print((t1-t0)/(t2-t1))

def test_CRS_product_large(m, n):

	sparse_matrix = np.zeros((m, n))
	# create a tridiagonal matrix with random integers
	np.fill_diagonal(sparse_matrix, np.random.randint(0, 3, m))
	np.fill_diagonal(sparse_matrix[:,1:], np.random.randint(0, 3, m-1))
	np.fill_diagonal(sparse_matrix[1:,:], np.random.randint(0, 3, m-1))

	A_CRS = CRS(sparse_matrix)
	t0 = time.clock()
	left_mult = np.random.rand(m)
	right_mult = np.random.rand(n)
	true_right_val = np.dot(sparse_matrix, right_mult)
	true_left_val = np.dot(left_mult, sparse_matrix)
	t1 = time.clock()
	right_val = A_CRS * right_mult
	left_val = CRS.multiply(left_mult, A_CRS)
	t2 = time.clock()

	# print(true_right_val, right_val)
	assert np.allclose(true_right_val, right_val)
	assert np.allclose(true_left_val, left_val)
	# print("%dx%d: %.2f" % (m, n, (t1-t0)/(t2-t1)))

def test_CRS_matrix_vector_product():
	test_CRS_product_known()
	array_dims = [2, 5, 10, 23, 50, 100, 200, 500, 1000]
	for m in array_dims:
		for n in array_dims:
			test_CRS_product_large(m, n)

def run_tests():
	test_inner_product()
	test_matrix_vector_product()
	test_CRS_matrix()
	test_CRS_matrix_vector_product()

if __name__ == "__main__":
	try:
		run_tests()
		print("All tests passed!")
	except Exception as e:
		print(e)
