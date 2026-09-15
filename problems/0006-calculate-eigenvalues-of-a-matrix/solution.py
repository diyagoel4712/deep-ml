import numpy as np

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:

	# # method 1
	# return np.linalg.eigvals(np.array(matrix)).tolist()

	# method 2
	arr = np.array(matrix)
	# trace = np.trace(arr)
	# det = np.linalg.det(arr)
	trace = arr[0,0] + arr[1,1]
	det = arr[0,0]*arr[1,1] - arr[0,1]*arr[1,0]
	val1 = (trace + np.sqrt(trace**2 - 4*det))/2
	val2 = (trace - np.sqrt(trace**2 - 4*det))/2
	return [val1, val2]

