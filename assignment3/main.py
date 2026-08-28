import random
import numpy as np
import matplotlib.pyplot as plt


DATA_FILE = "noisy_22.txt"
SEED = 42

MAX_DEGREE = 30

xs = []
ys = []

with open(DATA_FILE, "r") as f:
    for line in f:
        if line.strip():
            x, y = map(float, line.split())
            xs.append(x)
            ys.append(y)

xs = np.array(xs)
ys = np.array(ys)

indices = list(range(len(xs)))

random.seed(SEED)
random.shuffle(indices)

n = len(xs)

n_train = int(0.70 * n)
n_test = int(0.15 * n)

train_idx = indices[:n_train]
test_idx = indices[n_train:n_train + n_test]
val_idx = indices[n_train + n_test:]

x_train = xs[train_idx]
y_train = ys[train_idx]

x_test = xs[test_idx]
y_test = ys[test_idx]

x_val = xs[val_idx]
y_val = ys[val_idx]

print(
    f"train={len(x_train)}  "
    f"test={len(x_test)}  "
    f"val={len(x_val)}"
)

#main hero fam
def fit_polynomial(x, y, degree):
    A = np.array([
        [value ** p for p in range(degree + 1)]
        for value in x
    ])
    w = np.linalg.inv(A.T @ A) @ A.T @ y
    return w

def predict(x, w):

    degree = len(w) - 1

    X = np.array([
        [value ** p for p in range(degree + 1)]
        for value in x
    ])
    return X @ w

def mse(y_actual, y_pred):

    return np.mean(
        (y_actual - y_pred) ** 2
    )

# DEGREES 1 → 30
degrees = []
test_errors = []
val_errors = []

models = {}


print()
print(
    f"{'degree':>6} | "
    f"{'test MSE':>12} | "
    f"{'validation MSE':>16}"
)

print("-" * 42)


for degree in range(1, MAX_DEGREE + 1):

    try:

        w = fit_polynomial(
            x_train,
            y_train,
            degree
        )

        test_pred = predict(
            x_test,
            w
        )

        val_pred = predict(
            x_val,
            w
        )

        test_mse = mse(
            y_test,
            test_pred
        )

        val_mse = mse(
            y_val,
            val_pred
        )

        degrees.append(degree)
        test_errors.append(test_mse)
        val_errors.append(val_mse)

        models[degree] = w

        print(
            f"{degree:>6} | "
            f"{test_mse:>12.3f} | "
            f"{val_mse:>16.3f}"
        )

    except np.linalg.LinAlgError:

        print(
            f"{degree:>6} | matrix inversion failed"
        )

best_index = np.argmin(val_errors)

best_degree = degrees[best_index]

best_w = models[best_degree]

final_test_pred = predict(
    x_test,
    best_w
)

final_test_mse = mse(
    y_test,
    final_test_pred
)

print()
print("chosen degree =", best_degree)

print(
    "weights =",
    [round(float(x), 4) for x in best_w]
)

print(
    "final test MSE =",
    round(final_test_mse, 3)
)

x_line = np.linspace(
    min(xs),
    max(xs),
    1000
)



# DEGREE 1 dgram

y_line = predict(
    x_line,
    models[1]
)
plt.figure(figsize=(10, 6))
plt.scatter(
    xs,
    ys,
    s=10,
    alpha=0.25,
    label="data"
)
plt.plot(
    x_line,
    y_line,
    linewidth=3,
    label="degree 1"
)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Polynomial Regression - Degree 1")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "degree_1.png",
    dpi=150
)
plt.show()

# OPTIMAL Diagr

y_line = predict(
    x_line,
    best_w
)
plt.figure(figsize=(10, 6))

plt.scatter(
    xs,
    ys,
    s=10,
    alpha=0.25,
    label="data"
)

plt.plot(
    x_line,
    y_line,
    linewidth=3,
    label=f"degree {best_degree}"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title(
    f"Polynomial Regression - Optimal Degree {best_degree}"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "optimal_degree.png",
    dpi=150
)

plt.show()

# HIGHEST DEGREE Diagram


y_line = predict(
    x_line,
    models[MAX_DEGREE]
)

plt.figure(figsize=(10, 6))

plt.scatter(
    xs,
    ys,
    s=10,
    alpha=0.25,
    label="data"
)

plt.plot(
    x_line,
    y_line,
    linewidth=3,
    label=f"degree {MAX_DEGREE}"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title(
    f"Polynomial Regression - Degree {MAX_DEGREE}"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "highest_degree.png",
    dpi=150
)

plt.show()
plot_degrees = []
plot_errors = []

for degree, error in zip(
    degrees,
    val_errors
):

    if degree <= 21:

        plot_degrees.append(degree)
        plot_errors.append(error)


plt.figure(figsize=(10, 6))

plt.plot(
    plot_degrees,
    plot_errors,
    linewidth=2
)

plt.xlabel(
    "Polynomial Degree"
)

plt.ylabel(
    "Validation MSE"
)

plt.title(
    "Polynomial Degree vs Validation Error"
)

plt.xticks(
    range(1, 22)
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "mse_vs_degree_1_to_21.png",
    dpi=150
)

plt.show()


print()
print("saved:")
print("  degree_1.png")
print("  optimal_degree.png")
print("  highest_degree.png")
print("  mse_vs_degree_1_to_21.png")
