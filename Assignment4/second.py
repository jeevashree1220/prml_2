import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu1 = 2
mu2 = 5
sigma = 1

N = 1000

# Generate two Gaussian datasets
D1 = np.random.normal(mu1, sigma, N)
D2 = np.random.normal(mu2, sigma, N)

# Combine them into 2-D data points
X = np.column_stack((D1, D2))

# Mean vector
mu = np.array([mu1, mu2])

print("Mean vector:")
print(mu)

print("\nFirst 5 data points:")
print(X[:5])


# Covariance matrix
Sigma = np.cov(X, rowvar=False)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(Sigma)

print("\nCovariance matrix:")
print(Sigma)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)

# Case 1: Isotropic covariance
Sigma = sigma**2 * np.eye(2)

# Constant density curve
c = 5

theta = np.linspace(0, 2 * np.pi, 500)

x1 = mu[0] + sigma * np.sqrt(c) * np.cos(theta)
x2 = mu[1] + sigma * np.sqrt(c) * np.sin(theta)

# Plot
plt.figure(figsize=(7, 7))

plt.scatter(X[:, 0], X[:, 1], s=8, alpha=0.3)
plt.plot(x1, x2, linewidth=2)

# Mean
plt.scatter(mu[0], mu[1], marker='x', s=80)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Case 1: Isotropic Covariance")
plt.axis("equal")
plt.grid(True)

plt.show()


# Case 2: Diagonal covariance

sigma11 = 2
sigma22 = 0.5

Sigma = np.array([
    [sigma11**2, 0],
    [0, sigma22**2]
])

# Generate new data
Y = mu.reshape(2, 1) + np.sqrt(Sigma) @ (X.T - mu.reshape(2, 1))
Y = Y.T

# Covariance estimated from generated data
Sigma_est = np.cov(Y, rowvar=False)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(Sigma_est)

print("\nCase 2 Covariance Matrix:")
print(Sigma)

print("\nEstimated Covariance Matrix:")
print(Sigma_est)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)

# Case 2: Constant density ellipse

c = 5

theta = np.linspace(0, 2 * np.pi, 500)

x1 = mu[0] + sigma11 * np.sqrt(c) * np.cos(theta)
x2 = mu[1] + sigma22 * np.sqrt(c) * np.sin(theta)

plt.figure(figsize=(7, 7))

plt.scatter(Y[:, 0], Y[:, 1], s=8, alpha=0.3)
plt.plot(x1, x2, linewidth=2)

plt.scatter(mu[0], mu[1], marker='x', s=80)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Case 2: Diagonal Covariance")
plt.axis("equal")
plt.grid(True)

plt.show()


# Case 3: Full covariance

Sigma = np.array([
    [4, 1.5],
    [1.5, 1]
])

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

print("\nCase 3 Covariance Matrix:")
print(Sigma)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)

# Matrix square root of Sigma

Sigma_sqrt = (
    eigenvectors
    @ np.diag(np.sqrt(eigenvalues))
    @ eigenvectors.T
)

# Generate new data
Y = mu.reshape(2, 1) + Sigma_sqrt @ (X.T - mu.reshape(2, 1))
Y = Y.T

# Estimated covariance
Sigma_est = np.cov(Y, rowvar=False)

print("\nEstimated Covariance Matrix:")
print(Sigma_est)

# Case 3: Constant density ellipse

c = 5

theta = np.linspace(0, 2 * np.pi, 500)

circle = np.array([
    np.cos(theta),
    np.sin(theta)
])

curve = (
    mu.reshape(2, 1)
    + np.sqrt(c) * Sigma_sqrt @ circle
)

plt.figure(figsize=(7, 7))

plt.scatter(Y[:, 0], Y[:, 1], s=8, alpha=0.3)
plt.plot(curve[0], curve[1], linewidth=2)

plt.scatter(mu[0], mu[1], marker='x', s=80)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Case 3: Full Covariance")
plt.axis("equal")
plt.grid(True)
plt.show()


# Case 4: Two Gaussian classes

N = 1000

# Class 1
mu_1 = np.array([2, 5])
Sigma_1 = np.array([
    [4, 1.5],
    [1.5, 1]
])

# Class 2
mu_2 = np.array([7, 8])
Sigma_2 = np.array([
    [4, 1.5],
    [1.5, 1]
])

# Generate standard Gaussian data
X1 = np.random.randn(N, 2)
X2 = np.random.randn(N, 2)

# Transform to required Gaussian distributions
eigval1, eigvec1 = np.linalg.eigh(Sigma_1)
Sigma_sqrt1 = eigvec1 @ np.diag(np.sqrt(eigval1)) @ eigvec1.T

eigval2, eigvec2 = np.linalg.eigh(Sigma_2)
Sigma_sqrt2 = eigvec2 @ np.diag(np.sqrt(eigval2)) @ eigvec2.T

Y1 = mu_1 + X1 @ Sigma_sqrt1.T
Y2 = mu_2 + X2 @ Sigma_sqrt2.T

# Plot the two classes
plt.figure(figsize=(8, 7))

plt.scatter(Y1[:, 0], Y1[:, 1], s=8, alpha=0.4, label="Class ω1")
plt.scatter(Y2[:, 0], Y2[:, 1], s=8, alpha=0.4, label="Class ω2")

plt.scatter(mu_1[0], mu_1[1], marker='x', s=100)
plt.scatter(mu_2[0], mu_2[1], marker='x', s=100)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Case 4: Two Gaussian Classes")
plt.axis("equal")
plt.grid(True)
plt.legend()

plt.show()
