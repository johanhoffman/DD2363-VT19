import numpy as np

A = np.array([
	[1, 0, 2, 0],
	[3, 4, 0, 0],
	[0, 0, 5, 2],
	[0, 0, 0, 6]
])

val = np.array([1, 2, 3, 4, 5, 2, 6])
col_idx = np.array([0, 2, 0, 1, 2, 3, 3])
row_ptr = np.array([0, 2, 4, 6, 7])

m, n = A.shape
x = np.array([1, 2, 3, 4])
b = np.zeros(A.shape[0])

for i in range(m):
	# print()
	for j in range(row_ptr[i], row_ptr[i+1]):
		# print(val[j], x[col_idx[j]])
		b[i] += val[j]*x[col_idx[j]]

print(b)
print(np.dot(A, x))