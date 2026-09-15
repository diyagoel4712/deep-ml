import numpy as np

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:

	# # method 1
	# return np.linalg.eigvals(np.array(matrix))

	# method 2
	arr = np.array(matrix)
	trace = np.trace(arr)
	det = np.linalg.det(arr)
	val1 = (trace + np.sqrt(trace**2 - 4*det))/2
	val2 = (trace - np.sqrt(trace**2 - 4*det))/2
	return [val1.round(), val2.round()]

