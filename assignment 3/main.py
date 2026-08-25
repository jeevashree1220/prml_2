import random 
import math 
import matplotlib.pyplot as plt 

DATA_FILE = r"D:\programming\snu chennai\PRML\LINEAR REGRESSION\noisy_22.txt" 

SEED = 42  # fixed seed -> reproducible split across every run/report 
# ========================================================================= 
# PART A - HAND-WRITTEN MATH PRIMITIVES 
# ========================================================================= 
  
def mean(values): 
    """Arithmetic mean of a list of numbers. Replaces numpy.mean.""" 
    return sum(values) / len(values) 
  
def std(values): 
    """Population standard deviation. Replaces numpy.std.""" 
    m = mean(values) 
    variance = sum((v - m) ** 2 for v in values) / len(values) 
    return math.sqrt(variance) 


def mse(y_true, y_pred): 
    """Mean Squared Error: (1/N) * sum((y_i - yhat_i)^2). 
    This is the E(w) quantity from the theory, evaluated on real data. 
    Replaces sklearn.metrics.mean_squared_error. 
    """ 
    n = len(y_true) 
    if n == 0: 
        raise ValueError("Cannot compute MSE on an empty split.") 
    total = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) 
    return total / n 
  
def transpose(A): 
    """Matrix transpose A^T. Replaces numpy's A.T.""" 
    rows, cols = len(A), len(A[0]) 
    return [[A[r][c] for r in range(rows)] for c in range(cols)] 
  
def matmul(A, B): 
    """Matrix-matrix product A @ B. Replaces numpy's A @ B / np.dot.""" 
    n, k = len(A), len(A[0]) 
    k2, m = len(B), len(B[0]) 
    if k != k2: 
        raise ValueError(f"Shape mismatch for matmul: {k} != {k2}") 
    result = [[0.0] * m for _ in range(n)] 
    for i in range(n): 
        for p in range(k): 
            a_ip = A[i][p] 
            if a_ip == 0.0: 
                continue 
            for j in range(m): 
                result[i][j] += a_ip * B[p][j] 
    return result 

def matvec(A, v): 
    """Matrix-vector product A @ v. Replaces numpy's A @ v.""" 
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))] 


def gaussian_eliminate_solve(A, b): 

    n = len(A) 

    # Build augmented matrix [A | b] as a fresh copy so we don't mutate inputs. 

    M = [row[:] + [b[i]] for i, row in enumerate(A)] 

  

    SINGULAR_THRESHOLD = 1e-12  # below this, we treat the pivot as zero 

    min_pivot_seen = float("inf") 

  

    for col in range(n): 

        # --- partial pivoting: find the row with the largest |value| in this column --- 

        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col])) 

        if abs(M[pivot_row][col]) < SINGULAR_THRESHOLD: 

            raise ValueError( 

                f"Matrix is numerically singular at column {col} " 

                f"(best available pivot magnitude = {abs(M[pivot_row][col]):.3e}). " 

                f"Likely cause: polynomial degree too high for the amount " 

                f"of (effectively distinct) training data." 

            ) 

        M[col], M[pivot_row] = M[pivot_row], M[col] 

        min_pivot_seen = min(min_pivot_seen, abs(M[col][col])) 

  

        # --- eliminate this column from all rows below --- 

        pivot_val = M[col][col] 

        for r in range(col + 1, n): 

            factor = M[r][col] / pivot_val 

            if factor == 0.0: 

                continue 

            for c in range(col, n + 1): 

                M[r][c] -= factor * M[col][c] 

  

    # --- back-substitution --- 

    w = [0.0] * n 

    for i in range(n - 1, -1, -1): 

        known_sum = sum(M[i][j] * w[j] for j in range(i + 1, n)) 

        w[i] = (M[i][n] - known_sum) / M[i][i] 

  

    return w, min_pivot_seen 

  

  

# ========================================================================= 

# PART B - DATA LOADING AND SPLITTING 

# ========================================================================= 

  

def load_data(path): 

    """ 

    Read the two-column space-separated (x, y) file. 

    Skips blank lines defensively (trailing newline at EOF, etc). 

    """ 

    xs, ys = [], [] 

    with open(path, "r") as f: 

        for line_no, line in enumerate(f, start=1): 

            line = line.strip() 

            if not line: 

                continue  # guard against blank/trailing lines 

            parts = line.split() 

            if len(parts) != 2: 

                raise ValueError(f"Malformed row at line {line_no}: {line!r}") 

            x_val, y_val = float(parts[0]), float(parts[1]) 

            xs.append(x_val) 

            ys.append(y_val) 

    return xs, ys 

  

  

def train_test_val_split(xs, ys, train_frac=0.70, test_frac=0.15, seed=SEED): 

    """ 

    a 70/15/15 split). 

    """ 

    n = len(xs) 

    indices = list(range(n)) 

    random.seed(seed) 

    random.shuffle(indices) 

  

    n_train = int(round(train_frac * n)) 

    n_test = int(round(test_frac * n)) 

    # Remainder goes to validation, so the three counts always sum to n 

    # exactly even when train_frac*n / test_frac*n don't divide evenly. 

  

    train_idx = indices[:n_train] 

    test_idx = indices[n_train:n_train + n_test] 

    val_idx = indices[n_train + n_test:] 

  

    def gather(idx_list): 

        return [xs[i] for i in idx_list], [ys[i] for i in idx_list] 

  

    return gather(train_idx), gather(test_idx), gather(val_idx) 

  

  

# ========================================================================= 

# PART C - POLYNOMIAL REGRESSION MODEL 

# ========================================================================= 

  

def build_design_matrix(xs, degree): 

    """ 

    Build the Vandermonde / polynomial feature (design) matrix: 

        row i = [1, x_i, x_i^2, ..., x_i^degree] 

    so that predictions are simply X @ w. 

    """ 

    return [[x ** power for power in range(degree + 1)] for x in xs] 

  

  

def fit_polynomial(x_train, y_train, degree, x_mean, x_std): 

    """ 



    """ 

    x_scaled = [(x - x_mean) / x_std for x in x_train] 

    X = build_design_matrix(x_scaled, degree) 

  

    if degree + 1 > len(x_train): 

        # More parameters than training points => X^T X is guaranteed 

        # rank-deficient (singular), not just poorly conditioned. This 

        # is a hard mathematical limit worth failing loudly on. 

        raise ValueError( 

            f"degree={degree} needs {degree + 1} parameters but only " 

            f"{len(x_train)} training points are available -- system is " 

            f"guaranteed singular." 

        ) 

  

    Xt = transpose(X) 

    A = matmul(Xt, X)          # X^T X 

    b = matvec(Xt, y_train)    # X^T y 

    w, min_pivot = gaussian_eliminate_solve(A, b) 

    return w, min_pivot 

  

  

def predict(x_values, w, x_mean, x_std): 

    """Apply a fitted polynomial to new x values (same scaling as training).""" 

    x_scaled = [(x - x_mean) / x_std for x in x_values] 

    X = build_design_matrix(x_scaled, len(w) - 1) 

    return matvec(X, w) 

  

  

# ========================================================================= 
# PART D - CORE ASSIGNMENT 
# ========================================================================= 

  

def run_core_assignment(xs, ys): 

    print("=" * 70) 

    print("PART D: CORE ASSIGNMENT") 

    print("=" * 70) 

  

    (x_train, y_train), (x_test, y_test), (x_val, y_val) = \ 

        train_test_val_split(xs, ys, train_frac=0.70, test_frac=0.15) 

  

    print(f"Split sizes -> train: {len(x_train)}, test: {len(x_test)}, " 

          f"validation: {len(x_val)}  (total {len(xs)})") 

  

    # Scaling statistics computed from TRAIN only, reused everywhere else. 

    x_mean, x_std = mean(x_train), std(x_train) 

  

    # Candidate degrees kept in a numerically safe range (see PART E for 

    # what happens once you push degree much higher than this). 

    candidate_degrees = list(range(1, 16)) 

  

    best_degree = None 

    best_test_mse = float("inf") 

    best_w = None 

  

    print("\nFitting candidate degrees on TRAIN, scoring on TEST:") 

    print(f"{'degree':>6} | {'train MSE':>12} | {'test MSE':>12}") 

    for degree in candidate_degrees: 

        w, _ = fit_polynomial(x_train, y_train, degree, x_mean, x_std) 

        train_pred = predict(x_train, w, x_mean, x_std) 

        test_pred = predict(x_test, w, x_mean, x_std) 

        train_mse = mse(y_train, train_pred) 

        test_mse = mse(y_test, test_pred) 

        print(f"{degree:>6} | {train_mse:>12.3f} | {test_mse:>12.3f}") 

  

        if test_mse < best_test_mse: 

            best_test_mse = test_mse 

            best_degree = degree 

            best_w = w 

  

    # "Choose the parameters of the polynomial that fit the test data set." 

    print(f"\n-> Selected degree (lowest TEST MSE): M = {best_degree}") 

    print(f"   Fitted weights (w0 ... w{best_degree}):") 

    print("   " + ", ".join(f"{coef:.4f}" for coef in best_w)) 

  

    # "Report the results on validation set." 

    val_pred = predict(x_val, best_w, x_mean, x_std) 

    val_mse = mse(y_val, val_pred) 

    print(f"\n-> FINAL REPORTED RESULT (on VALIDATION set): " 

          f"MSE = {val_mse:.3f}") 

    print("=" * 70 + "\n") 

  

    return { 

        "x_train": x_train, "y_train": y_train, 

        "x_test": x_test, "y_test": y_test, 

        "x_val": x_val, "y_val": y_val, 

        "x_mean": x_mean, "x_std": x_std, 

        "best_degree": best_degree, "best_w": best_w, 

        "val_mse": val_mse, 

    } 

  

  

# ========================================================================= 

# PART E - BEYOND THE ASSIGNMENT (kept separate from the core result above) 


#   E1. How does fit quality change from underfitting to a good fit to 

#       the point where numerical conditioning starts to matter? 

#   E2. Visual confirmation: what do the underfit / good-fit / unstable 

#       curves actually look like against the data? 

# ========================================================================= 

  

# These four degrees were not picked arbitrarily -- a preliminary sweep 

# on this exact dataset showed: 

#   M=1  : clear underfitting (barely better than a flat line) 

#   M=8  : the sharp drop in error -- this is roughly where the true 

#          underlying structure gets captured 

#   M=20 : error has fully plateaued (more flexibility buys nothing) and 

#          the smallest pivot the solver sees during elimination has 

#          already dropped by >1000x compared to low degrees 

#   M=80 : error is still similar to M=20, BUT the solver's pivot 

#          magnitudes swing across ~9 orders of magnitude while solving 

#          -- the fit "succeeds" numerically but is no longer trustworthy 

STUDY_DEGREES = [1, 8, 20, 80] 

  

  

def run_beyond_experiments(xs, ys): 

    print("=" * 70) 

    print("PART E: BEYOND THE ASSIGNMENT") 

    print("=" * 70) 

  

    # Re-split with the SAME seed as the core section so results are 

    # directly comparable to Part D rather than testing on a different 

    # random partition. 

    (x_train, y_train), (x_test, y_test), (x_val, y_val) = \ 

        train_test_val_split(xs, ys, train_frac=0.70, test_frac=0.15) 

    x_mean, x_std = mean(x_train), std(x_train) 

  

    print("\n--- E1: degree vs. train/test/val error, plus solver conditioning ---") 

    print(f"{'degree':>6} | {'train MSE':>10} | {'test MSE':>10} | " 

          f"{'val MSE':>10} | {'min pivot seen':>16}") 

  

    fitted_models = {} 

    for degree in STUDY_DEGREES: 

        w, min_pivot = fit_polynomial(x_train, y_train, degree, x_mean, x_std) 

        train_mse = mse(y_train, predict(x_train, w, x_mean, x_std)) 

        test_mse = mse(y_test, predict(x_test, w, x_mean, x_std)) 

        val_mse = mse(y_val, predict(x_val, w, x_mean, x_std)) 

        fitted_models[degree] = w 

        print(f"{degree:>6} | {train_mse:>10.2f} | {test_mse:>10.2f} | " 

              f"{val_mse:>10.2f} | {min_pivot:>16.6e}") 

  



  

    print("\n--- E2: visualizing the fitted curves against the data ---") 

    plot_fitted_curves(x_train, y_train, fitted_models, x_mean, x_std) 

    print("Saved plot to fitted_curves.png") 

    print("=" * 70 + "\n") 

  

  

def plot_fitted_curves(x_train, y_train, fitted_models, x_mean, x_std): 

    """ 

    Overlay each studied degree's fitted curve on top of the raw (noisy) 

    data. matplotlib is used here ONLY to draw pixels -- every number 

    being plotted (the fitted curve's y-values) came from our own 

    from-scratch `predict()`, not from any library regression call. 

    """ 

    plt.figure(figsize=(9, 6)) 

  

    # Light scatter of a subsample of training points (plotting all 7000 

    # would just be visual noise / slow rendering). 

    sample_idx = list(range(0, len(x_train), max(1, len(x_train) // 800))) 

    plt.scatter([x_train[i] for i in sample_idx], 

                [y_train[i] for i in sample_idx], 

                s=8, alpha=0.3, color="gray", label="training data (sample)") 

  

    x_line = sorted(x_train)  # smooth curve needs sorted x 

    colors = {1: "tab:red", 8: "tab:green", 20: "tab:orange", 80: "tab:purple"} 

    for degree, w in fitted_models.items(): 

        y_line = predict(x_line, w, x_mean, x_std) 

        plt.plot(x_line, y_line, color=colors.get(degree, "black"), 

                  linewidth=1.5, label=f"degree {degree}") 

  

    plt.xlabel("x") 

    plt.ylabel("y") 

    plt.title("Polynomial fits of varying degree vs. noisy data") 

    plt.legend() 

    plt.ylim(min(y_train) - 20, max(y_train) + 20)  # keep high-degree wiggles from blowing up the axis 

    plt.tight_layout() 

    plt.savefig("fitted_curves.png", dpi=150) 

    plt.close() 

  

  

# ========================================================================= 

# MAIN 

# ========================================================================= 

  

if __name__ == "__main__": 

    xs, ys = load_data(DATA_FILE) 

    core_results = run_core_assignment(xs, ys) 

    run_beyond_experiments(xs, ys) 

 

 
 
