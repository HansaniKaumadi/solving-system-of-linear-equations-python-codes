import numpy as np

def is_positive_definite(matrix):
    if not np.allclose(matrix, matrix.T):
        raise ValueError("The matrix is not symmetric")

    n = matrix.shape[0]
    for i in range(1, n + 1):
        principal_submatrix = matrix[:i, :i]
        if np.linalg.det(principal_submatrix) <= 0:
            return False
    return True

# Example usage
matrix = np.array([[2, -1, 0],
                   [-1, 2, -1],
                   [0, -1, 2]])

print(is_positive_definite(matrix))