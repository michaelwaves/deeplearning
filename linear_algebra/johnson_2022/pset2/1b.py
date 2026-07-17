import torch

A = [[2, -2, 0, 1, 1],
     [2, -1, 1, -2, 1],
     [0, 3, 4, -3, 1],
     [0, 0, -2, -5, -3],
     [0, 0, 0, 7, 3]
     ]
A = torch.tensor(A, dtype=torch.float32)
LU = torch.linalg.lu(A)
breakpoint()
