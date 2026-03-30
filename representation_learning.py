import torch
import matplotlib.pyplot as plt

n = 500
x1 = torch.randn(n)
x2 = 3*x1 + 0.5*torch.randn(n)

X = torch.stack([x1,x2], dim=1)

X = X-X.mean(dim=0)
cov = X.T @X /n

eigvals, eigvecs = torch.linalg.eigh(cov)

idx = torch.argsort(eigvals, descending=True)
eigvals = eigvals[idx]
eigvecs = eigvecs[:,idx]

w = eigvecs[:,0]
X_proj = X@w.unsqueeze(1)
X_recon = X_proj@w.unsqueeze(0)

plt.scatter(X[:,0],X[:,1], alpha=0.3)
plt.plot(
    [-3*w[0], 3*w[0]],
    [-3*w[1], 3*w[1]],
    linewidth=3
)

plt.title("First principal component")
plt.savefig("pca.png")
breakpoint()