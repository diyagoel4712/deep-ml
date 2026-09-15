import numpy as np

def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    a = np.array(a)
    b = np.array(b)

    if a.shape[1] != b.shape[0]:
        return -1
    # c = np.matmul(a, b)
    c = [[np.dot(row,b[:,i]) for i in range(b.shape[1])] for row in a]
    return c