```python
import random
import math
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = "noisy_22.txt"   # must sit next to this script
SEED = 42                    # fixed seed -> reproducible split


# ---------------------------------------------------------------- math ---

def mse(y_true, y_pred):
    """Mean squared error."""
    if not y_true:
        raise ValueError("empty split - cannot compute MSE")

    return sum(
        (y_true[i] - y_pred[i]) ** 2
        for i in range(len(y_true))
    ) / len(y_true)


def transpose(A):
    """Matrix transpose (replaces A.T)."""
    return [
        [A[r][c] for r in range(len(A))]
        for c in range(len(A[0]))
    ]


def matmul(A, B):
    """Matrix-matrix multiply (replaces A @ B)."""
    n, k, m = len(A), len(A[0]), len(B[0])

    result = [[0.0] * m for _ in range(n)]

    for i in range(n):
        for p in range(k):

            if A[i][p] == 0.0:
                continue

            for j in range(m):
                result[i][j] += A[i][p] * B[p][j]

    return result


def matvec(A, v):
    """Matrix-vector multiply (replaces A @ v)."""
    return [
        sum(A[i][j] * v[j] for j in range(len(v)))
        for i in range(len(A))
    ]


# -------------------------------------------------------- pseudo inverse ---

def solve_with_pinv(A, b):
    """
    Solve A w = b using the Moore-Penrose pseudoinverse.

    Mathematically:

        w = A^+ b
    """

    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)

    A_pinv = np.linalg.pinv(A_np)

    w = A_pinv @ b_np

    return w.tolist()


# ---------------------------------------------------------------- data ---

def load_data(path):
    """Read space-separated (x, y) file."""

    xs, ys = [], []

    with open(path, "r") as f:

        for line_no, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 2:
                raise ValueError(
                    f"malformed row at line {line_no}: {line!r}"
                )

            x, y = float(parts[0]), float(parts[1])

            xs.append(x)
            ys.append(y)

    return xs, ys


def train_test_val_split(
    xs,
    ys,
    train_frac=0.70,
    test_frac=0.15,
    seed=SEED
):
    """
    Shuffle then split into train/test/validation.
    """

    n = len(xs)

    idx = list(range(n))

    random.seed(seed)
    random.shuffle(idx)

    n_train = int(round(train_frac * n))
    n_test = int(round(test_frac * n))

    train_idx = idx[:n_train]

    test_idx = idx[
        n_train:n_train + n_test
    ]

    val_idx = idx[
        n_train + n_test:
    ]

    pick = lambda ids: (
        [xs[i] for i in ids],
        [ys[i] for i in ids]
    )

    return (
        pick(train_idx),
        pick(test_idx),
        pick(val_idx)
    )


# ------------------------------------------------------------- model -----

def build_design_matrix(xs, degree):
    """
    Vandermonde/design matrix.

    For degree = 3:

        X =
        [1, x, x², x³]
    """

    return [
        [x ** p for p in range(degree + 1)]
        for x in xs
    ]


def fit_polynomial(x_train, y_train, degree):
    """
    Fit polynomial using the normal equation:

        (X^T X) w = X^T y

    solved using the Moore-Penrose pseudoinverse:

        w = (X^T X)^+ X^T y

    IMPORTANT:
    Raw x values are used.
    NO normalization.
    NO standardization.
    """

    if degree + 1 > len(x_train):
        raise ValueError(
            f"degree={degree} needs {degree + 1} points, "
            f"only {len(x_train)} available"
        )

    # ---------------------------------------------------------
    # 1. Build X
    # ---------------------------------------------------------

    X = build_design_matrix(
        x_train,
        degree
    )

    # ---------------------------------------------------------
    # 2. X^T
    # ---------------------------------------------------------

    Xt = transpose(X)

    # ---------------------------------------------------------
    # 3. X^T X
    # ---------------------------------------------------------

    XtX = matmul(
        Xt,
        X
    )

    # ---------------------------------------------------------
    # 4. X^T y
    # ---------------------------------------------------------

    Xty = matvec(
        Xt,
        y_train
    )

    # ---------------------------------------------------------
    # 5. Moore-Penrose pseudoinverse
    #
    # w = (X^T X)^+ X^T y
    # ---------------------------------------------------------

    w = solve_with_pinv(
        XtX,
        Xty
    )

    return w


def predict(x_values, w):
    """Predict y using the fitted polynomial."""

    X = build_design_matrix(
        x_values,
        len(w) - 1
    )

    return matvec(
        X,
        w
    )


# -------------------------------------------------------------- main -----

def run():

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    xs, ys = load_data(
        DATA_FILE
    )

    # ---------------------------------------------------------
    # Split data
    # ---------------------------------------------------------

    (x_tr, y_tr), \
    (x_te, y_te), \
    (x_va, y_va) = train_test_val_split(
        xs,
        ys
    )

    print(
        f"train={len(x_tr)}  "
        f"test={len(x_te)}  "
        f"val={len(x_va)}"
    )

    # ---------------------------------------------------------
    # Try polynomial degrees 1 → 15
    # ---------------------------------------------------------

    best_degree = None
    best_w = None
    best_test_mse = float("inf")

    print(
        f"{'degree':>6} | {'test MSE':>10}"
    )

    for degree in range(1, 16):

        # Fit polynomial
        w = fit_polynomial(
            x_tr,
            y_tr,
            degree
        )

        # Predict test data
        test_predictions = predict(
            x_te,
            w
        )

        # Calculate test error
        test_mse = mse(
            y_te,
            test_predictions
        )

        print(
            f"{degree:>6} | "
            f"{test_mse:>10.3f}"
        )

        # Keep best degree
        if test_mse < best_test_mse:

            best_degree = degree
            best_w = w
            best_test_mse = test_mse

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    val_predictions = predict(
        x_va,
        best_w
    )

    val_mse = mse(
        y_va,
        val_predictions
    )

    print(
        f"\nchosen degree = {best_degree}"
    )

    print(
        "weights =",
        [round(c, 4) for c in best_w]
    )

    print(
        f"validation MSE = {val_mse:.3f}"
    )

    # =========================================================
    # VISUALIZATION
    # =========================================================

    # Create a smooth x-axis for drawing the fitted polynomial.

    x_min = min(xs)
    x_max = max(xs)

    num_points = 500

    x_line = [
        x_min
        + (x_max - x_min)
        * i / (num_points - 1)
        for i in range(num_points)
    ]

    # Evaluate the fitted polynomial
    # at those 500 x-values.

    y_line = predict(
        x_line,
        best_w
    )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    # Original data
    plt.scatter(
        xs,
        ys,
        label="Original data",
        s=20
    )

    # Fitted polynomial
    plt.plot(
        x_line,
        y_line,
        label=f"Fitted polynomial (degree={best_degree})",
        linewidth=2
    )

    # Training points
    plt.scatter(
        x_tr,
        y_tr,
        label="Training data",
        s=15
    )

    # Test points
    plt.scatter(
        x_te,
        y_te,
        label="Test data",
        s=25
    )

    # Validation points
    plt.scatter(
        x_va,
        y_va,
        label="Validation data",
        s=25
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(
        f"Polynomial Regression — "
        f"Best Degree = {best_degree}"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    run()
```
