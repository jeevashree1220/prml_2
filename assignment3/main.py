
import random
import math
 
DATA_FILE = "noisy_22.txt"   # must sit next to this script
SEED = 42                    # fixed seed -> reproducible split
 
 
# ---------------------------------------------------------------- math ---
 
def mean(v):
    """Custom mean (replaces numpy.mean)."""
    return sum(v) / len(v)
 
 
def std(v):
    """Custom standard deviation (replaces numpy.std)."""
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
 
 
def mse(y_true, y_pred):
    """Mean squared error (replaces sklearn's mean_squared_error)."""
    if not y_true:
        raise ValueError("empty split - cannot compute MSE")
    return sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))) / len(y_true)
 
 
def transpose(A):
    """Matrix transpose (replaces A.T)."""
    return [[A[r][c] for r in range(len(A))] for c in range(len(A[0]))]
 
 
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
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]
 
 
def gaussian_eliminate_solve(A, b):
    """
    Solve A w = b via Gaussian elimination with partial pivoting,
    then back-substitution. Replaces numpy.linalg.solve.
 
    Edge case handled: if the best available pivot in a column is
    (numerically) zero, the system is singular - raise clearly instead
    of dividing by ~0 and returning garbage weights.
    """
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]  # augmented [A|b]
    SINGULAR_TOL = 1e-12
 
    for col in range(n):
        # pick the largest-magnitude entry in this column as the pivot row
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot_row][col]) < SINGULAR_TOL:
            raise ValueError(
                f"singular system at column {col}: degree too high for "
                f"the amount of (effectively distinct) training data"
            )
        M[col], M[pivot_row] = M[pivot_row], M[col]
 
        # eliminate this column from every row below the pivot
        for r in range(col + 1, n):
            factor = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]
 
    # back-substitution: solve from the last row upward
    w = [0.0] * n
    for i in range(n - 1, -1, -1):
        known = sum(M[i][j] * w[j] for j in range(i + 1, n))
        w[i] = (M[i][n] - known) / M[i][i]
    return w
 
 
# ---------------------------------------------------------------- data ---
 
def load_data(path):
    """Read space-separated (x, y) file. Skips blank lines (EOF newline etc)."""
    xs, ys = [], []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"malformed row at line {line_no}: {line!r}")
            x, y = float(parts[0]), float(parts[1])
            xs.append(x)
            ys.append(y)
    return xs, ys
 
 
def train_test_val_split(xs, ys, train_frac=0.70, test_frac=0.15, seed=SEED):
    """
    Shuffle then split into train/test/validation.
    Shuffling matters: this data is sorted by x, so a plain slice would
    give validation a different x-range than train.
    Remainder after train+test goes to validation, so counts always sum to n.
    """
    n = len(xs)
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
 
    n_train = int(round(train_frac * n))
    n_test = int(round(test_frac * n))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:n_train + n_test]
    val_idx = idx[n_train + n_test:]
 
    pick = lambda ids: ([xs[i] for i in ids], [ys[i] for i in ids])
    return pick(train_idx), pick(test_idx), pick(val_idx)
 
 
# ------------------------------------------------------------- model -----
 
def build_design_matrix(xs, degree):
    """Vandermonde matrix: row i = [1, x_i, x_i^2, ..., x_i^degree]."""
    return [[x ** p for p in range(degree + 1)] for x in xs]
 
 
def fit_polynomial(x_train, y_train, degree, x_mean, x_std):
    """
    Least-squares fit via normal equations: (X^T X) w = X^T y.
 
    x is standardized with TRAIN mean/std before building the design
    matrix - required here, not optional: x ranges to +-100, so x^10
    alone is ~1e20 and wrecks the solver's conditioning without it.
 
    Edge case: more parameters than training points guarantees a
    singular system - fail clearly instead of solving something invalid.
    """
    if degree + 1 > len(x_train):
        raise ValueError(
            f"degree={degree} needs {degree+1} points, only {len(x_train)} available"
        )
    x_scaled = [(x - x_mean) / x_std for x in x_train]
    X = build_design_matrix(x_scaled, degree)
    Xt = transpose(X)
    A = matmul(Xt, X)
    b = matvec(Xt, y_train)
    return gaussian_eliminate_solve(A, b)
 
 
def predict(x_values, w, x_mean, x_std):
    """Apply fitted polynomial to new x values, using the same scaling."""
    x_scaled = [(x - x_mean) / x_std for x in x_values]
    X = build_design_matrix(x_scaled, len(w) - 1)
    return matvec(X, w)
 
 
# -------------------------------------------------------------- main -----
 
def run():
    xs, ys = load_data(DATA_FILE)
    (x_tr, y_tr), (x_te, y_te), (x_va, y_va) = train_test_val_split(xs, ys)
    print(f"train={len(x_tr)}  test={len(x_te)}  val={len(x_va)}")
 
    x_mean, x_std = mean(x_tr), std(x_tr)  # scaling stats from TRAIN only
 
    best_degree, best_w, best_test_mse = None, None, float("inf")
    print(f"{'degree':>6} | {'test MSE':>10}")
    for degree in range(1, 16):                       # candidate degrees
        w = fit_polynomial(x_tr, y_tr, degree, x_mean, x_std)
        test_mse = mse(y_te, predict(x_te, w, x_mean, x_std))
        print(f"{degree:>6} | {test_mse:>10.3f}")
        if test_mse < best_test_mse:                   # pick by TEST error
            best_degree, best_w, best_test_mse = degree, w, test_mse
 
    val_mse = mse(y_va, predict(x_va, best_w, x_mean, x_std))  # report on VAL
    print(f"\nchosen degree = {best_degree}")
    print("weights =", [round(c, 4) for c in best_w])
    print(f"validation MSE = {val_mse:.3f}")
 
 
if __name__ == "__main__":
    run()
 
