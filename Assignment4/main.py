"""
2-D Gaussian data, covariance matrices, constant density curves, decision boundaries.
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
N = 1000
MU = np.array([2.0, 3.0])          # mu1, mu2 stacked into the mean vector

def gen_isotropic(mu, sigma2, n):
    """D1, D2: two independent 1-D Gaussians -> x = [x1, x2]^T."""
    d1 = mu[0] + np.sqrt(sigma2) * rng.standard_normal(n)
    d2 = mu[1] + np.sqrt(sigma2) * rng.standard_normal(n)
    return np.column_stack([d1, d2])


def sqrtm_sym(C):
    """Sigma^(1/2) via eigen-decomposition (C is symmetric PSD)."""
    w, V = np.linalg.eigh(C)
    return V @ np.diag(np.sqrt(w)) @ V.T


def transform(X, mu, C):
    """Y = mu + Sigma^(1/2) (X - mu)  --- eq. (1) in the problem sheet."""
    return mu + (X - mu) @ sqrtm_sym(C).T


def estimate(X):
    """ML/unbiased estimates of mean vector and covariance matrix."""
    mu_hat = X.mean(axis=0)
    Xc = X - mu_hat
    C_hat = (Xc.T @ Xc) / (len(X) - 1)
    return mu_hat, C_hat


def eig_sorted(C):
    """Eigenvalues descending + matching eigenvectors as columns."""
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    V *= np.sign(V[np.abs(V).argmax(axis=0), [0, 1]])   # fix sign for readability
    return w, V


def density_ellipse(mu, C, c, npts=400):
    """Locus of (x-mu)^T C^-1 (x-mu) = c^2  ->  mu + c * V diag(sqrt(lam)) u."""
    w, V = eig_sorted(C)
    t = np.linspace(0, 2 * np.pi, npts)
    u = np.vstack([np.cos(t), np.sin(t)])
    return (mu[:, None] + c * V @ np.diag(np.sqrt(w)) @ u).T


# ---------- plotting for Q1-Q3 ----------

def analyse(X, title, fname):
    mu_hat, C_hat = estimate(X)
    w, V = eig_sorted(C_hat)

    print(f"\n=== {title} ===")
    print("mean vector      :", np.round(mu_hat, 4))
    print("covariance matrix:\n", np.round(C_hat, 4))
    print("eigenvalues      :", np.round(w, 4))
    print("eigenvectors     :\n", np.round(V, 4))
    print(f"semi-major axis (1-sigma) = {np.sqrt(w[0]):.4f}  "
          f"semi-minor axis = {np.sqrt(w[1]):.4f}")
    print("axis ratio sqrt(l1/l2) =", np.round(np.sqrt(w[0] / w[1]), 4))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(X[:, 0], X[:, 1], s=6, alpha=0.3, color="steelblue", label="data")

    for c in (1, 2, 3):                       # 1,2,3 Mahalanobis-sigma contours
        E = density_ellipse(mu_hat, C_hat, c)
        ax.plot(E[:, 0], E[:, 1], lw=1.4, label=f"$c={c}$")

    colors = ["crimson", "darkgreen"]
    names = ["major", "minor"]
    for k in range(2):
        v = V[:, k] * np.sqrt(w[k])           # eigenvector scaled to 1-sigma length
        ax.annotate("", xy=mu_hat + v, xytext=mu_hat,
                    arrowprops=dict(arrowstyle="-|>", lw=2, color=colors[k]))
        ax.text(*(mu_hat + 1.08 * v),
                f"{names[k]}: $\\sqrt{{\\lambda_{k+1}}}$={np.sqrt(w[k]):.2f}",
                color=colors[k], fontsize=9)

    ax.plot(*mu_hat, "k+", ms=12)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(title); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)
    return mu_hat, C_hat, w, V


# ---------- Q4: two classes and their decision boundary ----------

def g_quadratic(x1, x2, mu1, C1, p1, mu2, C2, p2):
    """g(x) = ln p(x|w1)P(w1) - ln p(x|w2)P(w2)  (Bayes discriminant)."""
    P = np.dstack([x1, x2])

    def term(mu, C, p):
        Ci = np.linalg.inv(C)
        d = P - mu
        m = np.einsum("...i,ij,...j->...", d, Ci, d)     # Mahalanobis^2
        return -0.5 * m - 0.5 * np.log(np.linalg.det(C)) + np.log(p)

    return term(mu1, C1, p1) - term(mu2, C2, p2)


def two_class(mu1, C1, mu2, C2, title, fname, p1=0.5, p2=0.5, n=500):
    Y1 = transform(gen_isotropic(mu1, 1.0, n), mu1, C1)
    Y2 = transform(gen_isotropic(mu2, 1.0, n), mu2, C2)

    lo = np.minimum(Y1.min(0), Y2.min(0)) - 2
    hi = np.maximum(Y1.max(0), Y2.max(0)) + 2
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 400),
                         np.linspace(lo[1], hi[1], 400))
    G = g_quadratic(gx, gy, mu1, C1, p1, mu2, C2, p2)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.contourf(gx, gy, np.sign(G), levels=[-2, 0, 2],
                colors=["#fde7e7", "#e7f0fd"], alpha=0.8)
    ax.scatter(Y1[:, 0], Y1[:, 1], s=6, alpha=0.4, color="steelblue", label="$\\omega_1$")
    ax.scatter(Y2[:, 0], Y2[:, 1], s=6, alpha=0.4, color="indianred", label="$\\omega_2$")
    ax.contour(gx, gy, G, levels=[0], colors="k", linewidths=2)

    for mu, C, col in ((mu1, C1, "navy"), (mu2, C2, "darkred")):
        E = density_ellipse(mu, C, 2)
        ax.plot(E[:, 0], E[:, 1], color=col, lw=1.2, ls="--")
        ax.plot(*mu, "+", color=col, ms=12)

    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(title); ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)

    # empirical training error of the Bayes rule
    err = (g_quadratic(Y1[:, 0], Y1[:, 1], mu1, C1, p1, mu2, C2, p2) < 0).mean()
    err += (g_quadratic(Y2[:, 0], Y2[:, 1], mu1, C1, p1, mu2, C2, p2) > 0).mean()
    print(f"{title}: misclassification rate = {err / 2:.3f}")


def main():
    # Q1 - isotropic  C = sigma^2 I
    X = gen_isotropic(MU, 1.0, N)
    analyse(X, "Q1  Isotropic  $C=\\sigma^2 I$", "q1_isotropic.png")

    # Q2 - diagonal
    C_diag = np.array([[4.0, 0.0], [0.0, 1.0]])
    analyse(transform(X, MU, C_diag),
            "Q2  Diagonal  $C=\\mathrm{diag}(4,1)$", "q2_diagonal.png")

    # Q3 - full
    C_full = np.array([[4.0, 1.8], [1.8, 1.5]])
    analyse(transform(X, MU, C_full),
            "Q3  Full  $\\sigma_{12}\\neq 0$", "q3_full.png")

    # Q4 - decision boundaries
    I = np.eye(2)
    two_class(np.array([1.0, 1.0]), I, np.array([5.0, 5.0]), I,
              "Q4a  $C_1=C_2=\\sigma^2 I$ : linear, perpendicular bisector",
              "q4a_linear_bisector.png")

    Cs = np.array([[3.0, 1.5], [1.5, 1.0]])
    two_class(np.array([1.0, 1.0]), Cs, np.array([5.0, 4.0]), Cs,
              "Q4b  $C_1=C_2$ (full) : linear, not perpendicular",
              "q4b_linear_rotated.png")

    two_class(np.array([2.0, 2.0]), np.array([[3.0, 0.0], [0.0, 3.0]]),
              np.array([4.0, 3.0]), np.array([[1.0, -0.6], [-0.6, 0.6]]),
              "Q4c  $C_1\\neq C_2$ : quadratic boundary",
              "q4c_quadratic.png")

    two_class(np.array([0.0, 0.0]), np.eye(2) * 0.5,
              np.array([0.0, 0.0]), np.eye(2) * 4.0,
              "Q4d  same mean, different spread : circular boundary",
              "q4d_circular.png")


if __name__ == "__main__":
    main()
