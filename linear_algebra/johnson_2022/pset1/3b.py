import numpy as np

B = np.random.randn(5, 5)
y = np.random.randn(5)

A = B-np.identity(5)/2
b = (B+np.identity(5)/2)@y

x = np.linalg.inv(A) @ b
x = np.linalg.solve(A, b)

check = np.allclose(B@(x-y), (x+y)/2, rtol=0.00001)
breakpoint()
