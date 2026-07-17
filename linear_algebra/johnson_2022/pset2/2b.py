import numpy as np

A = np.array([[1, -3, 0, 3],
              [0, 1, 1, 2],
              [0, 0, 1, -1],
              [0, 0, 0, 1]
              ])

B = np.array([
    [-3, -1, 1, -2],
    [-3, 0, 0, -3],
    [1, 3, 3, -3],
    [2, 3, 2, 0]
])

C = A@np.linalg.inv(A@B)

breakpoint()
