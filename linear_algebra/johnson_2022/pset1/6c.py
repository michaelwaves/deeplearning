import numpy as np

A = [[1, 6, - 3],
     [- 2, 3, 4],
     [1, 0, - 2]]
A = np.array(A)
B = [[7, 0],
     [3, -2],
     [0, 1]]
B = np.array(B)
X = np.linalg.solve(A, B)
breakpoint()
