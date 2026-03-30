import torch

# A = torch.tensor([[1, 3], [2, 4]], dtype=torch.float16)
# x = torch.tensor([1.0, 1.0], dtype=torch.float16, requires_grad=True)
# print(A@x)

# x1 = torch.tensor([1, 1])
# B = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])

# print(x1@B)

# y = (x**2).sum()
# y.backward()

# print(x.grad)

y = torch.tensor(1.3)
x_in = torch.tensor([1., 1.], requires_grad=True)
W = torch.tensor([[1., 2.], [3., 4.], [1., 1.1]], requires_grad=True)
x_out = W@x_in
h = torch.relu(x_out)
W_decode = torch.tensor([1., 1., 1.], requires_grad=True)
y_predict = W_decode@h
l2_loss = torch.nn.functional.mse_loss(y_predict, y)
breakpoint()
