import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	def verify_invertible(matrix):
		if matrix.shape[0] == matrix.shape[1]:
			if np.linalg.det(matrix) != 0:
				return True
		return False
	A = np.array(A)
	S = np.array(S)
	T = np.array(T)
	if not (verify_invertible(T) and verify_invertible(S)):
		return -1
	T_inv = np.linalg.inv(T)
	return np.matmul(np.matmul(T_inv, A), S).tolist()