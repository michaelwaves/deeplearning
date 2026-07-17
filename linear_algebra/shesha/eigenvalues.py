import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=100)

# A small rectangular matrix: 5 samples, 3 features
np.random.seed(42)
X = np.random.randn(5, 3)
print(f"X shape: {X.shape}")
print(f"X =\n{X}")

# Singular value decomposition
U, S, Vt = np.linalg.svd(X, full_matrices=False)
print(f"\nSingular values of X: {S}")

# Eigenvalues of X^T X (a 3x3 matrix)
XtX = X.T @ X
eigvals_XtX = np.linalg.eigvalsh(XtX)[::-1]  # sorted descending
print(f"Eigenvalues of X^T X: {eigvals_XtX}")

# The claim: eigenvalues of X^T X = singular values of X, squared
print(f"Singular values squared: {S**2}")
print(f"Match? {np.allclose(S**2, eigvals_XtX)}")

# Same story for XX^T
XXt = X @ X.T
eigvals_XXt = np.linalg.eigvalsh(XXt)[::-1]
print(f"\nEigenvalues of X X^T (top 3): {eigvals_XXt[:3]}")
print(f"Match? {np.allclose(S**2, eigvals_XXt[:3])}")
