import numpy as np

def scaled_partial_pivoting_gauss_elimination(A, b):
    n = len(b)
    scale = np.max(np.abs(A), axis=1)  # Scale factors
    row_interchanges = []

    for k in range(n-1):
        # Scaled partial pivoting
        ratios = np.abs(A[k:n, k]) / scale[k:n]
        pivot_row = np.argmax(ratios) + k

        if k != pivot_row:
            A[[k, pivot_row]] = A[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]
            scale[[k, pivot_row]] = scale[[pivot_row, k]]
            row_interchanges.append((k, pivot_row))

        for i in range(k+1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:n] -= factor * A[k, k:n]
            b[i] -= factor * b[k]

    augmented_matrix = np.hstack((A, b.reshape(-1, 1)))
    return augmented_matrix, row_interchanges

# Coefficient matrix A
A = np.array([[3, -13, 9, 3],
              [-6, 4, 1, -18],
              [6, -2, 2, 4],
              [12, -8, 6, 10]], dtype=float)

# Constant vector b
b = np.array([-19, -34, 16, 26], dtype=float)

# Solve the system and get the final augmented matrix and row interchanges
augmented_matrix, row_interchanges = scaled_partial_pivoting_gauss_elimination(A, b)

# Print the row interchanges
print("Row interchanges:", row_interchanges)

# Print the final augmented matrix
print("Final augmented matrix:\n", augmented_matrix)