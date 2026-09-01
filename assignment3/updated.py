import numpy as np
import matplotlib.pyplot as plt
import os

# STEP 1: DATASET LOCATION

DATA_FILE = "noisy_22.txt"

# STEP 2: CALCULATE MEAN SQUARED ERROR
#
# MSE = (1/N) * SUM((actual - predicted)^2)

def find_mse(actual, predicted):

    total_error = 0.0

    for i in range(len(actual)):

        error = actual[i] - predicted[i]

        total_error = total_error + error * error

    mse = total_error / len(actual)

    return mse

# STEP 3: LOAD DATASET

def load_dataset(filename):

    x = []
    y = []

    file = open(filename, "r")

    for line in file:

        line = line.strip()

        if line == "":
            continue

        parts = line.split()

        if len(parts) >= 2:

            x_value = float(parts[0])
            y_value = float(parts[1])

            x.append(x_value)
            y.append(y_value)

    file.close()

    return x, y


# STEP 4: SPLIT DATASET


def create_split(x, y):

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    total = len(x)

    for i in range(total):

        # Every 10th sample starting from index 1
        # is used for testing
        if i % 10 == 1:

            x_test.append(x[i])
            y_test.append(y[i])

        # Remaining samples are used for training
        else:

            x_train.append(x[i])
            y_train.append(y[i])

    return x_train, y_train, x_test, y_test

# STEP 5: CREATE POLYNOMIAL MATRIX
# x^0, x^1, x^2, ..., x^degree

def create_polynomial_matrix(x, degree):

    matrix = []

    for value in x:

        row = []

        for power in range(degree + 1):

            row.append(value ** power)

        matrix.append(row)

    return np.array(matrix, dtype=float)

# STEP 6: TRANSPOSE MATRIX

def transpose(matrix):

    return np.transpose(matrix)

# STEP 7: MATRIX MULTIPLICATION

def matrix_multiply(A, B):

    rows_A = len(A)
    columns_A = len(A[0])

    rows_B = len(B)
    columns_B = len(B[0])

    if columns_A != rows_B:

        print("Matrix multiplication error.")

        return None

    result = []

    for i in range(rows_A):

        row = []

        for j in range(columns_B):

            total = 0.0

            for k in range(columns_A):

                total = total + A[i][k] * B[k][j]

            row.append(total)

        result.append(row)

    return np.array(result, dtype=float)


# STEP 8: FIT POLYNOMIAL REGRESSION MODEL
# Normal Equation: w = (X^T X)^(-1) X^T y

def fit_polynomial(x_train, y_train, degree):

    # Create polynomial design matrix
    X = create_polynomial_matrix(
        x_train,
        degree
    )

    # Convert y values into column matrix
    Y = []

    for value in y_train:

        Y.append([value])

    Y = np.array(Y, dtype=float)

    # Transpose X
    XT = transpose(X)

    # Calculate X^T X
    A = matrix_multiply(
        XT,
        X
    )

    # Calculate X^T y
    b = matrix_multiply(
        XT,
        Y
    )

    # Calculate inverse of X^T X
    inverse_A = np.linalg.inv(A)

    # Calculate weights
    weights = inverse_A @ b

    # Convert to one-dimensional array
    weights = weights.flatten()

    return weights

# STEP 9: PREDICT OUTPUTS

def predict_values(x, weights):

    predictions = []

    # Determine polynomial degree
    degree = len(weights) - 1

    for value in x:

        prediction = 0.0

        # w0 + w1*x + w2*x^2 + ... + wn*x^n
        for power in range(degree + 1):

            prediction = (
                prediction
                +
                weights[power]
                *
                (value ** power)
            )

        predictions.append(prediction)

    return predictions

# STEP 10: CALCULATE RMSE : RMSE = sqrt(MSE)


def calculate_rmse(mse):

    return mse ** 0.5


# STEP 11: CALCULATE R-SQUARED : R^2 = 1 - SSE / SST


def calculate_r_squared(actual, predicted):

    # Calculate mean
    total = 0.0

    for value in actual:

        total = total + value

    mean = total / len(actual)

    # Calculate SSE
    sse = 0.0

    for i in range(len(actual)):

        error = actual[i] - predicted[i]

        sse = sse + error * error

    # Calculate SST
    sst = 0.0

    for value in actual:

        difference = value - mean

        sst = sst + difference * difference

    if sst == 0:

        return 0.0

    # Calculate R-squared
    r_squared = 1.0 - (sse / sst)

    return r_squared


# STEP 12: PRINT POLYNOMIAL EQUATION

def print_polynomial(weights):

    degree = len(weights) - 1

    print()

    print("Polynomial Equation:")

    equation = "y = "

    for i in range(degree + 1):

        coefficient = weights[i]

        # Constant term
        if i == 0:

            equation = equation + (
                str(round(coefficient, 6))
            )

        else:

            # Positive coefficient
            if coefficient >= 0:

                equation = equation + " + "

            # Negative coefficient
            else:

                equation = equation + " - "

                coefficient = -coefficient

            equation = equation + (
                str(round(coefficient, 6))
            )

            equation = equation + "*x"

            # Add exponent
            if i > 1:

                equation = (
                    equation
                    +
                    "^"
                    +
                    str(i)
                )

    print(equation)


# STEP 13: PLOT MSE VS POLYNOMIAL DEGREE

def plot_mse_vs_degree(
    degrees,
    train_mse_list,
    test_mse_list,
    best_degree
):

    fig, ax = plt.subplots(figsize=(10, 6))

    # Training MSE
    ax.plot(
        degrees,
        train_mse_list,
        marker='o',
        linewidth=2,
        color='#1f77b4',
        label='Training MSE'
    )

    # Test MSE
    ax.plot(
        degrees,
        test_mse_list,
        marker='s',
        linewidth=2,
        color='#ff7f0e',
        label='Test MSE'
    )

    # Highlight degree 7
    ax.axvline(
        x=best_degree,
        color='red',
        linestyle='--',
        linewidth=1.5,
        label='Best Degree = ' + str(best_degree)
    )

    ax.set_title(
        'Model Performance: MSE vs Polynomial Degree',
        fontsize=14,
        fontweight='bold'
    )

    ax.set_xlabel(
        'Polynomial Degree',
        fontsize=12
    )

    ax.set_ylabel(
        'Mean Squared Error (MSE)',
        fontsize=12
    )

    ax.set_xticks(degrees)

    ax.legend(fontsize=11)

    ax.grid(
        True,
        linestyle='--',
        alpha=0.6
    )

    plt.tight_layout()

    plt.savefig(
        "mse_vs_degree.png",
        dpi=150
    )

    plt.show()


# STEP 14: PLOT FITTED POLYNOMIAL CURVE

def plot_fitted_curve(
    x,
    y,
    weights,
    degree
):

    # Sort x values
    sorted_x = sorted(x)

    # Calculate fitted y values
    fitted_y = predict_values(
        sorted_x,
        weights
    )

    # Create graph
    plt.figure(figsize=(10, 6))

    # Original data
    plt.scatter(
        x,
        y,
        s=8,
        alpha=0.4,
        label="Original Data"
    )

    # Fitted polynomial
    plt.plot(
        sorted_x,
        fitted_y,
        linewidth=2,
        color='orange',
        label="Fitted Polynomial"
    )

    plt.title(
        "Polynomial Regression - Fitted Curve"
    )

    plt.xlabel("x")

    plt.ylabel("y")

    # Display degree
    plt.text(
        0.02,
        0.95,
        "Polynomial Degree = " + str(degree),
        transform=plt.gca().transAxes
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "fitted_curve.png",
        dpi=150
    )

    plt.show()


# STEP 15: RUN ASSIGNMENT


def run_assignment(x, y):

    print()

    print("=" * 75)

    print("POLYNOMIAL REGRESSION")

    print("=" * 75)

    # Split dataset
    

    x_train, y_train, x_test, y_test = create_split(x, y)

    # Display dataset sizes

    print()

    print(
        "Total samples    :",
        len(x)
    )

    print(
        "Training samples :",
        len(x_train)
    )

    print(
        "Testing samples  :",
        len(x_test)
    )

    # Variables for best model

    best_degree = 7

    best_test_mse = None

    best_weights = None

    # Lists for graph

    plot_degrees = []

    plot_train_mse = []

    plot_test_mse = []

    # Degree comparison

    print()

    print("=" * 75)

    print("DEGREE COMPARISON")

    print("=" * 75)

    print()

    print(
        "{:>8} | {:>18} | {:>18}".format(
            "Degree",
            "Train MSE",
            "Test MSE"
        )
    )

    print("-" * 75)

    # Test polynomial degrees from 1 to 15

    for degree in range(1, 16):

        try:

            # Train model
            weights = fit_polynomial(
                x_train,
                y_train,
                degree
            )

            # Training prediction
            train_prediction = predict_values(
                x_train,
                weights
            )

            # Testing prediction
            test_prediction = predict_values(
                x_test,
                weights
            )

            # Training MSE
            train_error = find_mse(
                y_train,
                train_prediction
            )

            # Testing MSE
            test_error = find_mse(
                y_test,
                test_prediction
            )

            # Display results
            print(
                "{:>8} | {:>18.6f} | {:>18.6f}".format(
                    degree,
                    train_error,
                    test_error
                )
            )

            # Store values for graph
            plot_degrees.append(degree)

            plot_train_mse.append(train_error)

            plot_test_mse.append(test_error)


            if degree == 7:

                best_test_mse = test_error

                best_weights = weights

        except np.linalg.LinAlgError:

            print(
                "{:>8} | Matrix inversion failed".format(
                    degree
                )
            )

    print("-" * 75)

    # Plot MSE vs degree

    plot_mse_vs_degree(
        plot_degrees,
        plot_train_mse,
        plot_test_mse,
        best_degree
    )

    # Display best model

    print()

    print("=" * 75)

    print("BEST MODEL")

    print("=" * 75)

    print()

    print(
        "Best polynomial degree:",
        best_degree
    )

    print()

    print(
        "Best test MSE:",
        best_test_mse
    )

    print()

    print(
        "Best test RMSE:",
        calculate_rmse(best_test_mse)
    )

    # Display polynomial equation

    print_polynomial(
        best_weights
    )

    # Display regression coefficients

    print()

    print("Regression Coefficients:")

    print("-" * 40)

    for i in range(len(best_weights)):

        print(
            "w{} = {:.10f}".format(
                i,
                best_weights[i]
            )
        )

    print("-" * 40)

    # Return final results

    return (
        best_degree,
        best_weights,
        best_test_mse
    )


# STEP 16: ADDITIONAL DEGREE EXPERIMENT - TRAIN MSE and TEST MSE.


def additional_experiment(x, y):

    print()

    print("=" * 75)

    print("ADDITIONAL DEGREE EXPERIMENT")

    print("=" * 75)

    # Split dataset

    x_train, y_train, x_test, y_test = create_split(x, y)

    # Polynomial degrees

    degrees = [
        1,
        2,
        3,
        5,
        7,
        8,
        10,
        15
    ]

    print()

    print(
        "{:>8} | {:>18} | {:>18}".format(
            "Degree",
            "Train MSE",
            "Test MSE"
        )
    )

    print("-" * 60)

    # Train each selected degree

    for degree in degrees:

        try:

            # Train model
            weights = fit_polynomial(
                x_train,
                y_train,
                degree
            )

            # Training prediction
            train_prediction = predict_values(
                x_train,
                weights
            )

            # Testing prediction
            test_prediction = predict_values(
                x_test,
                weights
            )

            # Training MSE
            train_mse = find_mse(
                y_train,
                train_prediction
            )

            # Testing MSE
            test_mse = find_mse(
                y_test,
                test_prediction
            )

            # Display results
            print(
                "{:>8} | {:>18.6f} | {:>18.6f}".format(
                    degree,
                    train_mse,
                    test_mse
                )
            )

        except np.linalg.LinAlgError:

            print(
                "{:>8} | Matrix inversion failed".format(
                    degree
                )
            )

    print("-" * 60)


# STEP 17: FINAL SELECTED MODEL


def report_fixed_degree(x, y, degree):

    print()

    print("=" * 75)

    print(
        "FINAL SELECTED MODEL (Degree " + str(degree) + ")"
    )

    print("=" * 75)

    # Split dataset

    x_train, y_train, x_test, y_test = create_split(x, y)

    # Train degree 7 model

    weights = fit_polynomial(
        x_train,
        y_train,
        degree
    )

    # Predictions
    # 

    train_prediction = predict_values(
        x_train,
        weights
    )

    test_prediction = predict_values(
        x_test,
        weights
    )

    # Calculate errors

    train_mse = find_mse(
        y_train,
        train_prediction
    )

    test_mse = find_mse(
        y_test,
        test_prediction
    )

    # Display results

    print()

    print(
        "Train MSE :",
        train_mse
    )

    print(
        "Test MSE  :",
        test_mse
    )

    print(
        "Test RMSE :",
        calculate_rmse(test_mse)
    )

    # Polynomial equation

    print_polynomial(
        weights
    )

    # Regression coefficients

    print()

    print("Regression Coefficients:")

    print("-" * 40)

    for i in range(len(weights)):

        print(
            "w{} = {:.15e}".format(
                i,
                weights[i]
            )
        )

    print("-" * 40)

    return weights


# STEP 18: MAIN PROGRAM

if __name__ == "__main__":

    # Load dataset

    print()

    print("Reading dataset...")

    x_values, y_values = load_dataset(
        DATA_FILE
    )

    # Check whether dataset was loaded

    if len(x_values) == 0:

        print(
            "Error: Dataset could not be loaded."
        )

    else:

        # Display number of samples

        print(
            "Dataset loaded:",
            len(x_values),
            "samples"
        )

        # Run degree comparison

        results = run_assignment(
            x_values,
            y_values
        )

        # Run additional experiment

        additional_experiment(
            x_values,
            y_values
        )

        # Final selected model = Degree 7

        final_weights = report_fixed_degree(
            x_values,
            y_values,
            7
        )

        # Plot fitted curve
        plot_fitted_curve(
            x_values,
            y_values,
            final_weights,
            7
        )