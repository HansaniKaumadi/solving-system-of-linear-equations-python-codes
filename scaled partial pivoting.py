import numpy as np

def scaled_partial_pivoting_gauss_elimination(A, b):
    n = len(b)
    scale = np.max(np.abs(A), axis=1)  # Scale factors

    # Forward Elimination
    for k in range(n-1):
        # Scaled partial pivoting
        ratios = np.abs(A[k:n, k]) / scale[k:n]
        pivot_row = np.argmax(ratios) + k

        if k != pivot_row:
            A[[k, pivot_row]] = A[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]
            scale[[k, pivot_row]] = scale[[pivot_row, k]]

        for i in range(k+1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:n] -= factor * A[k, k:n]
            b[i] -= factor * b[k]

    # Back Substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    return x

# Example usage
A = np.array([[4, 2, 5,3],
              [3, 2, 1,4],
              [5, -8, 6,3],
              [1,-1,2,1]], dtype=float)

b = np.array([-1, 1, 1,1], dtype=float)

solution = scaled_partial_pivoting_gauss_elimination(A, b)
print("Solution:", solution)