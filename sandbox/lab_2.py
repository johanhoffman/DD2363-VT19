import numpy as np

def normalize(v):
	return v / np.linalg.norm(v)

def norm(v):
	return np.linalg.norm(v)

# def gram_schmidt(A):
	# Q = np.zeros_like(A, dtype = np.float64)
	# Q[:,0] = normalize(A[:,0])
	# for j in range(1, A.shape[1]):
		# col = A[:,j].copy()
		# for i in range(j):
			# col -= np.dot(A[:,j], Q[:,i]) * Q[:,i]
		# Q[:,j] = normalize(col)
	# return Q

def g_s(A):
	V = A.copy()
	Q = np.zeros_like(A, dtype = np.float64)
	R = Q.copy()
	for i in range(A.shape[0]):
		r_ii = norm(V[:,i])
		R[i,i] = r_ii
		q_i = V[:,i] / r_ii
		for j in range(i+1, n):
			r_ij = np.dot(q_i, V[:,j])
			V[:,j] -= r_ij * q_i
			R[i,j] = r_ij
		Q[:,i] = q_i
	return Q, R

def invert_upper_triangular(U):
	pass


n = 2
A = np.arange(n*n, dtype = np.float64)
A.shape = (n,n)
print(np.dot(A, np.array([5, 6])))
# A = np.random.rand(n, n)
# print(A)
# Q2 = gram_schmidt(A)
Q, R = g_s(A)
print(Q)
print(R)
print("||A - Q*R||:", (A - np.dot(Q, R)).sum())
# print(Q2)
for i in range(n):
	for j in range(n):
		print(round(np.abs(np.dot(Q[:,i], Q[:,j])), 5))