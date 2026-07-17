import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
D = diags(
    [np.full(6, 1), np.full(6, 4), np.full(4, 7)],
    offsets=[0, 1, -2],
    shape=(6, 8)
).toarray()

D_sparse = diags(
    [np.full(4, -1), np.full(4, 1)],
    offsets=[0, 1],
    shape=(4, 5)

).toarray()


a, b = 0, 5
n = 100
delta_x = (b-a)/(n)
D_sparse_100 = diags(
    [np.full(99, -1/delta_x), np.full(99, 1/delta_x)],
    offsets=[0, 1],
    shape=(99, 100)
)

x = np.arange(a, b, delta_x)
print(len(x))
f = np.sin(x)
print(len(f))
plt.plot(x, f, 'r')
plt.plot(x[1:], D_sparse_100@f, 'b')
plt.savefig("approx.png")
breakpoint()
