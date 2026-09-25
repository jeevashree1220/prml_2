import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================

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


# ============================================================
# SAMPLE COVARIANCE
# ============================================================

Sigma_sample = np.cov(X, rowvar=False)

print("\nSample Covariance Matrix:")
print(Sigma_sample)


# ============================================================
# FUNCTION: MAHALANOBIS DISTANCE
# ============================================================

def mahalanobis_distance(points, mu, Sigma):

    Sigma_inv = np.linalg.inv(Sigma)

    diff = points - mu

    distances_squared = np.array([
        d.T @ Sigma_inv @ d
        for d in diff
    ])

    return np.sqrt(distances_squared)


# ============================================================
# FUNCTION: MAHALANOBIS DISTANCE GRID
# ============================================================

def create_mahalanobis_grid(mu, Sigma, x_range, y_range, N_grid=500):

    x1 = np.linspace(x_range[0], x_range[1], N_grid)
    x2 = np.linspace(y_range[0], y_range[1], N_grid)

    X1, X2 = np.meshgrid(x1, x2)

    # Convert grid into individual 2-D points
    grid_points = np.column_stack((
        X1.ravel(),
        X2.ravel()
    ))

    # Calculate Mahalanobis distance
    distances = mahalanobis_distance(
        grid_points,
        mu,
        Sigma
    )

    # Convert back to grid
    distances = distances.reshape(X1.shape)

    return X1, X2, distances


# ============================================================
# CASE 1: ISOTROPIC COVARIANCE
# ============================================================

Sigma = sigma**2 * np.eye(2)

print("\n========================================")
print("CASE 1: ISOTROPIC COVARIANCE")
print("========================================")

print("\nCovariance Matrix:")
print(Sigma)

# Mahalanobis distances of data
distances = mahalanobis_distance(X, mu, Sigma)

print("\nFirst 5 Mahalanobis distances:")
print(distances[:5])

# Create grid
X1, X2, M = create_mahalanobis_grid(
    mu,
    Sigma,
    (mu[0] - 3, mu[0] + 3),
    (mu[1] - 3, mu[1] + 3)
)

# Constant Mahalanobis distance
c = 5

plt.figure(figsize=(7, 7))

plt.scatter(
    X[:, 0],
    X[:, 1],
    s=8,
    alpha=0.3
)

# d_M^2 = c
plt.contour(
    X1,
    X2,
    M**2,
    levels=[c],
    linewidths=2
)

# Mean
plt.scatter(
    mu[0],
    mu[1],
    marker='x',
    s=80
)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Case 1: Isotropic Covariance")
plt.axis("equal")
plt.grid(True)

plt.show()


# ============================================================
# CASE 2: DIAGONAL COVARIANCE
# ============================================================

sigma11 = 2
sigma22 = 0.5

Sigma = np.array([
    [sigma11**2, 0],
    [0, sigma22**2]
])

print("\n========================================")
print("CASE 2: DIAGONAL COVARIANCE")
print("========================================")

print("\nCovariance Matrix:")
print(Sigma)

# Generate standard Gaussian data
Z = np.random.randn(N, 2)

# Generate Gaussian data with required covariance
Y = mu + Z @ np.linalg.cholesky(Sigma).T

# Mahalanobis distances
distances = mahalanobis_distance(Y, mu, Sigma)

print("\nFirst 5 Mahalanobis distances:")
print(distances[:5])

# Create grid
X1, X2, M = create_mahalanobis_grid(
    mu,
    Sigma,
    (mu[0] - 7, mu[0] + 7),
    (mu[1] - 3, mu[1] + 3)
)

# Constant Mahalanobis distance
c = 5

plt.figure(figsize=(7, 7))

plt.scatter(
    Y[:, 0],
    Y[:, 1],
    s=8,
    alpha=0.3
)

# d_M^2 = c
plt.contour(
    X1,
    X2,
    M**2,
    levels=[c],
    linewidths=2
)

# Mean
plt.scatter(
    mu[0],
    mu[1],
    marker='x',
    s=80
)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Case 2: Diagonal Covariance")
plt.axis("equal")
plt.grid(True)

plt.show()


# ============================================================
# CASE 3: FULL COVARIANCE
# ============================================================

Sigma = np.array([
    [4, 1.5],
    [1.5, 1]
])

print("\n========================================")
print("CASE 3: FULL COVARIANCE")
print("========================================")

print("\nCovariance Matrix:")
print(Sigma)

# Generate standard Gaussian data
Z = np.random.randn(N, 2)

# Generate Gaussian data with required covariance
Y = mu + Z @ np.linalg.cholesky(Sigma).T

# Mahalanobis distances
distances = mahalanobis_distance(Y, mu, Sigma)

print("\nFirst 5 Mahalanobis distances:")
print(distances[:5])

# Create grid
X1, X2, M = create_mahalanobis_grid(
    mu,
    Sigma,
    (mu[0] - 7, mu[0] + 7),
    (mu[1] - 5, mu[1] + 5)
)

# Constant Mahalanobis distance
c = 5

plt.figure(figsize=(7, 7))

plt.scatter(
    Y[:, 0],
    Y[:, 1],
    s=8,
    alpha=0.3
)

# d_M^2 = c
plt.contour(
    X1,
    X2,
    M**2,
    levels=[c],
    linewidths=2
)

# Mean
plt.scatter(
    mu[0],
    mu[1],
    marker='x',
    s=80
)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Case 3: Full Covariance")
plt.axis("equal")
plt.grid(True)

plt.show()


# ============================================================
# CASE 4: TWO GAUSSIAN CLASSES
# ============================================================

N = 1000

# ------------------------------------------------------------
# Class 1
# ------------------------------------------------------------

mu_1 = np.array([2, 5])

Sigma_1 = np.array([
    [4, 1.5],
    [1.5, 1]
])

# ------------------------------------------------------------
# Class 2
# ------------------------------------------------------------

mu_2 = np.array([7, 8])

Sigma_2 = np.array([
    [4, 1.5],
    [1.5, 1]
])

# Generate standard Gaussian data
Z1 = np.random.randn(N, 2)
Z2 = np.random.randn(N, 2)

# Generate Gaussian data
Y1 = mu_1 + Z1 @ np.linalg.cholesky(Sigma_1).T
Y2 = mu_2 + Z2 @ np.linalg.cholesky(Sigma_2).T


# ============================================================
# MAHALANOBIS DISTANCES FOR EACH CLASS
# ============================================================

d1 = mahalanobis_distance(
    Y1,
    mu_1,
    Sigma_1
)

d2 = mahalanobis_distance(
    Y2,
    mu_2,
    Sigma_2
)

print("\n========================================")
print("CASE 4: TWO GAUSSIAN CLASSES")
print("========================================")

print("\nFirst 5 Mahalanobis distances for Class 1:")
print(d1[:5])

print("\nFirst 5 Mahalanobis distances for Class 2:")
print(d2[:5])


# ============================================================
# PLOT TWO CLASSES
# ============================================================

plt.figure(figsize=(8, 7))

plt.scatter(
    Y1[:, 0],
    Y1[:, 1],
    s=8,
    alpha=0.4,
    label="Class ω1"
)

plt.scatter(
    Y2[:, 0],
    Y2[:, 1],
    s=8,
    alpha=0.4,
    label="Class ω2"
)

# Means
plt.scatter(
    mu_1[0],
    mu_1[1],
    marker='x',
    s=100
)

plt.scatter(
    mu_2[0],
    mu_2[1],
    marker='x',
    s=100
)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Case 4: Two Gaussian Classes")
plt.axis("equal")
plt.grid(True)
plt.legend()

plt.show()
